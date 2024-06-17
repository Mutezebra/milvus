package csv

import (
	"context"
	"encoding/csv"
	"fmt"
	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"go.uber.org/atomic"
	"io"
	"time"
)

type reader struct {
	ctx    context.Context
	cancel func()
	cm     storage.ChunkManager
	schema *schemapb.CollectionSchema

	fileSize  *atomic.Int64
	filePaths []string

	bufferSize int
	count      int64 // count of one batch
	comma      rune
	haveFields bool

	parsers []*Parser
	dataCh  chan *storage.InsertData
	future1 *conc.Future[struct{}]
	future2 *conc.Future[struct{}]
	tasks   chan int16
}

type Row = map[storage.FieldID]any

const (
	CommaKey      = "comma"
	HaveFieldsKey = "have_fields"
)

// NewReader initializes a new instance of reader with the provided parameters.
func NewReader(ctx context.Context, cm storage.ChunkManager, schema *schemapb.CollectionSchema, paths []string, bufferSize int, options map[string]string) (*reader, error) {
	count, err := estimateReadCountPerBatch(bufferSize, schema)
	if err != nil {
		return nil, err
	}
	child, cancel := context.WithCancel(ctx)

	reader := &reader{
		ctx:        child,
		cancel:     cancel,
		cm:         cm,
		schema:     schema,
		fileSize:   atomic.NewInt64(0),
		filePaths:  paths,
		bufferSize: bufferSize,
		count:      count,
		comma:      ',',
		haveFields: true,
		parsers:    make([]*Parser, len(paths)),
		dataCh:     make(chan *storage.InsertData, 1),
	}
	if err = reader.initConfig(options); err != nil {
		return nil, err
	}
	if err = reader.initParsers(); err != nil {
		return nil, err
	}
	reader.initFutures()
	return reader, nil
}

// Read returns the data or an error.
func (r *reader) Read() (*storage.InsertData, error) {
	select {
	case data := <-r.dataCh:
		return data, nil
	case <-r.inner():
		r.close()
		err := r.err()
		if err == nil {
			return nil, io.EOF
		}
		return nil, err
	}
}

func (r *reader) Size() (int64, error) {
	if size := r.fileSize.Load(); size != 0 {
		return size, nil
	}
	for i := 0; i < len(r.parsers); i++ {
		size, err := r.cm.Size(r.ctx, r.filePaths[i])
		if err != nil {
			return 0, err
		}
		r.fileSize.Add(size)
	}
	return r.fileSize.Load(), nil
}

func (r *reader) Close() {}

func (r *reader) initConfig(options map[string]string) error {
	for key, v := range options {
		switch key {
		case CommaKey:
			if len(v) != 1 {
				return merr.WrapErrImportFailed("the length of comma must be 1")
			}
			r.comma = rune(v[0])
		case HaveFieldsKey:
			if v == "TRUE" || v == "True" || v == "true" || v == "1" {
				r.haveFields = true
			} else if v == "FALSE" || v == "False" || v == "false" || v == "0" {
				r.haveFields = false
			} else {
				return merr.WrapErrImportFailed("the wrong value of key `have_fields`")
			}
		}
	}
	return nil
}

// initParsers initializes parsers for processing each file.
func (r *reader) initParsers() error {
	for i, path := range r.filePaths {
		cmReader, err := r.cm.Reader(r.ctx, path)
		if err != nil {
			return merr.WrapErrImportFailed(fmt.Sprintf("read csv file failed, path=%s, err=%s", path, err.Error()))
		}
		rd := csv.NewReader(cmReader)
		rd.Comma = r.comma

		if !r.haveFields {
			if r.parsers[i], err = r.noFields(rd); err != nil {
				return err
			}
			if _, err = cmReader.Seek(0, io.SeekStart); err != nil {
				return merr.WrapErrImportFailed(fmt.Sprintf("failed when seek file to start,error: %v", err))
			}
			rd = csv.NewReader(cmReader)
			rd.Comma = r.comma
			r.parsers[i].UpdateReader(rd)
			continue
		}
		if r.parsers[i], err = r.readFirstLine(rd); err != nil {
			return err
		}
	}
	return nil
}

// readFirstLine To check if staticField has any missing or duplicate entries, and whether DynamicField exists
func (r *reader) readFirstLine(csvReader *csv.Reader) (*Parser, error) {
	pr := NewParser(csvReader)
	fieldNames, err := csvReader.Read()
	if err != nil {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("failed to read first line from file, error: %v", err))
	}
	name2Index := make(map[string]int)
	for index, name := range fieldNames {
		if _, ok := name2Index[name]; ok {
			return nil, merr.WrapErrImportFailed(fmt.Sprintf("duplicated key is not allowed, key=%s", name))
		}
		name2Index[name] = index
	}
	for _, field := range r.schema.GetFields() {
		if field.GetIsDynamic() {
			continue
		}
		if field.GetIsPrimaryKey() && field.GetAutoID() {
			if _, ok := name2Index[field.GetName()]; ok {
				return nil, merr.WrapErrImportFailed(
					fmt.Sprintf("the primary key '%s' is auto-generated, no need to provide", field.GetName()))
			}
			continue
		}
		if _, ok := name2Index[field.GetName()]; !ok {
			return nil, merr.WrapErrImportFailed(
				fmt.Sprintf("value of field '%s' is missed", field.GetName()))
		}
		var dim int64
		if typeutil.IsVectorType(field.DataType) && !typeutil.IsSparseFloatVectorType(field.DataType) {
			dim, err = typeutil.GetDim(field)
			if err != nil {
				return nil, err
			}
		}
		pr.AddColParser(name2Index[field.GetName()], int(dim), field)
		delete(name2Index, field.GetName())
	}

	if len(name2Index) != 0 && !r.schema.GetEnableDynamicField() {
		return nil, merr.WrapErrImportFailed("this collection do not enable dynamic field")
	}
	if r.schema.EnableDynamicField {
		dynamicField := typeutil.GetDynamicField(r.schema)
		metaIndex := -1
		if index, ok := name2Index[dynamicField.GetName()]; ok {
			metaIndex = index
			delete(name2Index, dynamicField.GetName())
		}
		pr.AddDynamicFieldParser(metaIndex, name2Index, dynamicField)
	}
	return pr, nil
}

// noFields To make judgments when there are no fields
func (r *reader) noFields(csvReader *csv.Reader) (*Parser, error) {
	pr := NewParser(csvReader)
	result, err := csvReader.Read()
	if err != nil {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("failed to read first line from file, error: %v", err))
	}
	index := 0
	for _, field := range r.schema.GetFields() {
		if field.GetIsPrimaryKey() && field.GetAutoID() {
			continue
		}
		if field.GetIsDynamic() {
			dynamicField := typeutil.GetDynamicField(r.schema)
			pr.AddDynamicFieldParser(index, make(map[string]int), dynamicField)
			index++
			continue
		}
		var dim int64
		if typeutil.IsVectorType(field.DataType) && !typeutil.IsSparseFloatVectorType(field.DataType) {
			dim, err = typeutil.GetDim(field)
			if err != nil {
				return nil, err
			}
		}
		pr.AddColParser(index, int(dim), field)
		index++
	}
	if len(result) != index {
		if !r.schema.EnableDynamicField { // If it does not include dynamic fields, then it should throw an error directly
			if len(result) > index {
				return nil, merr.WrapErrImportFailed("there are too many fields in file")
			}
			return nil, merr.WrapErrImportFailed("dont`t have enough fields in file")
		}
		if len(result) != index-1 { // If dynamic fields are included, it is permissible for the file to not contain $meta, which will automatically be set to {}.
			return nil, merr.WrapErrImportFailed("don`t have enough fields in file")
		}
	}
	return pr, nil
}

// initFutures initializes concurrent processing tasks.
func (r *reader) initFutures() {
	r.future1 = conc.Go(r.startProcess)
	if len(r.parsers) > 1 {
		r.future2 = conc.Go(r.startProcess)
		r.tasks = make(chan int16, len(r.parsers)+2)
	} else {
		r.tasks = make(chan int16, len(r.parsers)+1)
	}
	for i := 0; i < len(r.parsers); i++ {
		r.tasks <- int16(i)
	}
	if len(r.parsers) > 1 {
		r.tasks <- -1
	}
	r.tasks <- -1
}

// startProcess handles the concurrent processing of CSV parsing tasks.
func (r *reader) startProcess() (struct{}, error) {
	for taskIndex := range r.tasks {
		if taskIndex == -1 {
			return struct{}{}, nil
		}
		if err := r.handlerParser(taskIndex); err != nil {
			return struct{}{}, err
		}
	}
	return struct{}{}, nil
}

// handlerParser responsible for processing the specified parser,
// which is used to continuously process and send the parsed data.
func (r *reader) handlerParser(i int16) error {
	parser := r.parsers[i]
	for {
		data, err := r.processBatch(parser)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}

		select {
		case <-r.ctx.Done():
			return nil
		default:
			r.dataCh <- data
		}
	}
}

// processBatch processes a single batch of records from the CSV file.
func (r *reader) processBatch(pr *Parser) (*storage.InsertData, error) {
	insertData, err := storage.NewInsertData(r.schema)
	if err != nil {
		return nil, err
	}
	var cnt int64
	var countNumber int
	for {
		if cnt >= r.count {
			cnt = 0
			if insertData.GetMemorySize() >= r.bufferSize {
				break
			}
		}
		row, err := pr.Parse()
		if err != nil {
			if errors.Is(err, io.EOF) && countNumber != 0 {
				return insertData, nil
			}
			return insertData, err
		}
		if err = insertData.Append(row); err != nil {
			return nil, err
		}
		countNumber++
		cnt++
	}
	return insertData, nil
}

func (r *reader) close() {
	r.cancel()
	go func() { time.Sleep(1 * time.Millisecond); close(r.dataCh) }()
	close(r.tasks)
	for _ = range r.dataCh {
	}
}

func (r *reader) inner() <-chan struct{} {
	if len(r.parsers) == 1 {
		return r.future1.Inner()
	}
	if r.future1.Done() {
		return r.future2.Inner()
	}
	if r.future2.Done() {
		return r.future1.Inner()
	}
	return r.future1.Inner()
}

func (r *reader) err() error {
	if len(r.parsers) == 1 {
		return r.future1.Err()
	}
	if r.future1.Done() {
		return r.future1.Err()
	}
	if r.future2.Done() {
		return r.future2.Err()
	}
	return nil
}

func estimateReadCountPerBatch(bufferSize int, schema *schemapb.CollectionSchema) (int64, error) {
	sizePerRecord, err := typeutil.EstimateMaxSizePerRecord(schema)
	if err != nil {
		return 0, err
	}
	if 1000*sizePerRecord <= bufferSize {
		return 1000, nil
	}
	return int64(bufferSize) / int64(sizePerRecord), nil
}
