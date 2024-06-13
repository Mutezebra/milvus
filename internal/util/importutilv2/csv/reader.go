package csv

import (
	"context"
	"encoding/csv"
	"fmt"
	"github.com/cockroachdb/errors"
	"io"
	"sync"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"go.uber.org/atomic"
)

type reader struct {
	ctx    context.Context
	cm     storage.ChunkManager
	schema *schemapb.CollectionSchema

	fileSize  *atomic.Int64
	filePaths []string
	r         *csv.Reader

	bufferSize int
	count      int64 // count of one batch

	parsers    []*Parser
	once       sync.Once
	leaveTasks *atomic.Int32
	dataCh     chan *storage.InsertData
	errCh      chan error
	cancel     func()
	wg         sync.WaitGroup
}

type Row = map[storage.FieldID]any

// NewReader initializes a new instance of reader with the provided parameters.
func NewReader(ctx context.Context, cm storage.ChunkManager, schema *schemapb.CollectionSchema, paths []string, bufferSize int) (*reader, error) {
	count, err := estimateReadCountPerBatch(bufferSize, schema)
	if err != nil {
		return nil, err
	}
	reader := &reader{
		ctx:        ctx,
		cm:         cm,
		schema:     schema,
		fileSize:   atomic.NewInt64(0),
		filePaths:  paths,
		bufferSize: bufferSize,
		count:      count,
		parsers:    make([]*Parser, len(paths)),
		leaveTasks: atomic.NewInt32(int32(len(paths))),
		dataCh:     make(chan *storage.InsertData),
		errCh:      make(chan error),
		cancel:     func() {},
	}
	if err = reader.initParsers(); err != nil {
		return nil, err
	}

	return reader, nil
}

// Read initiates reading from the CSV files and returns the data or an error.
func (r *reader) Read() (*storage.InsertData, error) {
	r.once.Do(func() {
		ctx, cancel := context.WithCancel(r.ctx)
		r.cancel = cancel
		for i := 0; i < len(r.filePaths); i++ {
			go r.read(ctx, i)
		}
		r.wg.Add(len(r.filePaths))
	})
	data := <-r.dataCh
	err := <-r.errCh
	time.Sleep(3 * time.Millisecond) // to prevent the defer function from not having been executed yet
	if errors.Is(err, io.EOF) {      // there are two scenarios: one is that a certain file has been completely read, and the other is that all files have been read. As long as the program executes normally, the final return will definitely be here, so there is no need to use defer r.close() to close the resources
		if r.leaveTasks.Load() == 0 {
			r.close()
			return data, err
		}
		return data, nil
	}
	if err != nil {
		r.close()
		return data, err
	}
	return data, nil
}

// read is a private method that manages reading and parsing data from a single file.
func (r *reader) read(ctx context.Context, i int) {
	defer func() { r.leaveTasks.Dec(); r.wg.Done() }()
	pr := r.parsers[i]
	for {
		select {
		case <-ctx.Done():
			return
		default:
			data, err := r.processBatch(pr)
			r.dataCh <- data
			r.errCh <- err
			if err != nil {
				return
			}
			continue
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

// close gracefully terminates all operations and closes channels.Non-idempotent.
func (r *reader) close() { //
	r.cancel()
	go func() { // in the event of an error, it is possible to receive data that was successfully read but not yet received, along with the error.
		for r.leaveTasks.Load() != 0 {
			_ = <-r.dataCh
			_ = <-r.errCh
		}
	}()
	r.wg.Wait()
	close(r.dataCh)
	close(r.errCh)
}

var Comma = '\x01'
var haveFields = false

// initParsers initializes parsers for processing each file.
func (r *reader) initParsers() error {
	for i, path := range r.filePaths {
		cmReader, err := r.cm.Reader(r.ctx, path)
		if err != nil {
			return merr.WrapErrImportFailed(fmt.Sprintf("read csv file failed, path=%s, err=%s", path, err.Error()))
		}
		rd := csv.NewReader(cmReader)
		rd.Comma = Comma
		r.r = rd

		if !haveFields {
			if r.parsers[i], err = r.noFields(); err != nil {
				return err
			}
			if _, err = cmReader.Seek(0, io.SeekStart); err != nil {
				return merr.WrapErrImportFailed(fmt.Sprintf("failed when seek file to start,error: %v", err))
			}
			rd = csv.NewReader(cmReader)
			rd.Comma = Comma
			r.parsers[i].UpdateReader(rd)
			continue
		}
		if r.parsers[i], err = r.readFirstLine(); err != nil {
			return err
		}
	}
	return nil
}

// readFirstLine To check if staticField has any missing or duplicate entries, and whether DynamicField exists
func (r *reader) readFirstLine() (*Parser, error) {
	pr := NewParser(r.r)
	fieldNames, err := r.r.Read()
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
func (r *reader) noFields() (*Parser, error) {
	pr := NewParser(r.r)
	result, err := r.r.Read()
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

func (r *reader) Size() (int64, error) {
	if size := r.fileSize.Load(); size != 0 {
		return size, nil
	}
	size, err := r.cm.Size(r.ctx, r.filePaths[0])
	if err != nil {
		return 0, err
	}
	r.fileSize.Store(size)
	return size, nil
}

func (r *reader) Close() {}

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
