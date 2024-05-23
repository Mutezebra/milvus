package csv

import (
	"context"
	"encoding/csv"
	"fmt"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"go.uber.org/atomic"
	"io"
	"sync"
)

type reader struct {
	ctx    context.Context
	cm     storage.ChunkManager
	schema *schemapb.CollectionSchema

	fileSize *atomic.Int64
	filePath string
	r        *csv.Reader

	bufferSize int
	count      int64 // count of one batch

	parser *Parser
	once   *sync.Once
}

type Row = map[storage.FieldID]any

func NewReader(ctx context.Context, cm storage.ChunkManager, schema *schemapb.CollectionSchema, path string, bufferSize int) (*reader, error) {
	cmReader, err := cm.Reader(ctx, path)
	if err != nil {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("read csv file failed, path=%s, err=%s", path, err.Error()))
	}
	r := csv.NewReader(cmReader)
	count, err := estimateReadCountPerBatch(bufferSize, schema)
	if err != nil {
		return nil, err
	}
	reader := &reader{
		ctx:        ctx,
		cm:         cm,
		schema:     schema,
		fileSize:   atomic.NewInt64(0),
		filePath:   path,
		r:          r,
		bufferSize: bufferSize,
		count:      count,
		parser:     NewParser(r),
		once:       &sync.Once{},
	}
	if err = reader.readFirstLine(); err != nil {
		return nil, err
	}
	return reader, nil
}

func (r *reader) Read() (*storage.InsertData, error) {
	insertData, err := storage.NewInsertData(r.schema)
	if err != nil {
		return nil, err
	}
	rowsNumber := 0
	var cnt int64
	for {
		if cnt >= r.count {
			cnt = 0
			if insertData.GetMemorySize() >= r.bufferSize {
				break
			}
		}
		row, err := r.parser.Parse()
		if err != nil {
			if err == io.EOF && rowsNumber != 0 {
				return insertData, nil
			}
			return nil, err
		}
		fmt.Printf("\n read one line \n")
		err = insertData.Append(row)
		if err != nil {
			return nil, err
		}
		fmt.Printf("insert data append successfully\n")
		rowsNumber++
		cnt++
	}
	fmt.Printf("\n the insert data is %v \n", insertData)
	return insertData, nil
}

func (r *reader) Size() (int64, error) {
	if size := r.fileSize.Load(); size != 0 {
		return size, nil
	}
	size, err := r.cm.Size(r.ctx, r.filePath)
	if err != nil {
		return 0, err
	}
	r.fileSize.Store(size)
	return size, nil
}

func (r *reader) Close() {}

// 判断staticField是否有缺失或者重复，以及DynamicField是否存在
func (r *reader) readFirstLine() error {
	fieldNames, err := r.r.Read()
	if err != nil {
		return merr.WrapErrImportFailed(fmt.Sprintf("failed to read first line from file, error: %v", err))
	}
	name2Index := make(map[string]int)
	for index, name := range fieldNames {
		if _, ok := name2Index[name]; ok {
			return merr.WrapErrImportFailed(fmt.Sprintf("duplicated key is not allowed, key=%s", name))
		}
		name2Index[name] = index
	}
	for _, field := range r.schema.GetFields() {
		if field.GetIsDynamic() {
			continue
		}
		if field.GetIsPrimaryKey() && field.GetAutoID() { // 主键并且是自动ID
			if _, ok := name2Index[field.GetName()]; ok {
				return merr.WrapErrImportFailed(
					fmt.Sprintf("the primary key '%s' is auto-generated, no need to provide", field.GetName()))
			}
			continue
		}
		if _, ok := name2Index[field.GetName()]; !ok { // 如果这个字段不在csv文件中
			return merr.WrapErrImportFailed(
				fmt.Sprintf("value of field '%s' is missed", field.GetName()))
		}
		// 这个字段在文件中
		dim := int64(0)
		if typeutil.IsVectorType(field.DataType) && !typeutil.IsSparseFloatVectorType(field.DataType) {
			dim, err = typeutil.GetDim(field)
			if err != nil {
				return err
			}
		}
		r.parser.AddColParser(name2Index[field.GetName()], int(dim), field)
		delete(name2Index, field.GetName())
	}

	if len(name2Index) != 0 && !r.schema.GetEnableDynamicField() {
		return merr.WrapErrImportFailed("this collection do not enable dynamic field")
	}
	if r.schema.EnableDynamicField {
		dynamicField := typeutil.GetDynamicField(r.schema)
		metaIndex := -1
		if index, ok := name2Index[dynamicField.GetName()]; ok {
			metaIndex = index
			delete(name2Index, dynamicField.GetName())
		}
		r.parser.AddDynamicFieldParser(metaIndex, name2Index, dynamicField)
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
