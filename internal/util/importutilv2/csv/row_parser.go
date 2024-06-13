package csv

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"io"
	"strconv"
	"strings"
)

type Parser struct {
	line    int
	parsers []parser // 只负责存储所有的parser
	r       *csv.Reader
}

// parser 抽象了 colParser 和 dynamicFieldParser，
// 他们一个负责处理相对应的staticField，一个负责处理动态字段
type parser interface {
	parse(records []string, line int) (any, error)
	getFieldID() int64
}

func NewParser(reader *csv.Reader) *Parser {
	return &Parser{
		line:    1,
		parsers: make([]parser, 0),
		r:       reader,
	}
}

func (p *Parser) Parse() (Row, error) {
	record, err := p.r.Read()
	if err != nil && err != io.EOF {
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("failed when read from csv file, error: %v", err))
	}
	if err != nil {
		return nil, err
	}
	p.line++
	row := make(map[storage.FieldID]any)
	for _, parser := range p.parsers {
		data, err := parser.parse(record, p.line)
		if err != nil {
			return nil, err
		}
		row[parser.getFieldID()] = data
	}
	return row, nil
}

// AddColParser 添加一个新的staticFieldParser
func (p *Parser) AddColParser(index int, dim int, schema *schemapb.FieldSchema) {
	cp := &colParser{
		index:       index,
		dim:         dim,
		fieldSchema: schema,
	}
	p.parsers = append(p.parsers, cp)
}

func (p *Parser) AddDynamicFieldParser(index int, name2index map[string]int, schema *schemapb.FieldSchema) {
	dcp := &dynamicFieldParser{
		index:       index,
		fieldSchema: schema,
		name2index:  name2index,
	}
	p.parsers = append(p.parsers, dcp)
}

func (p *Parser) UpdateReader(r *csv.Reader) {
	p.r = r
}

type colParser struct {
	index       int
	dim         int
	fieldSchema *schemapb.FieldSchema
}

func (p *colParser) wrapTypeError(v any, line int) error {
	field := p.fieldSchema
	return merr.WrapErrImportFailed(fmt.Sprintf("expected type '%s' for field '%s', got wrong format with value '%v' in line %d",
		field.GetDataType().String(), field.GetName(), v, line))
}

func (p *colParser) wrapTypeWithError(v any, line int, err error) error {
	field := p.fieldSchema
	return merr.WrapErrImportFailed(fmt.Sprintf("expected type '%s' for field '%s', got wrong format with value '%v' in line %d, error: %v",
		field.GetDataType().String(), field.GetName(), v, line, err))
}

func (p *colParser) wrapArrayValueTypeError(v any, eleType schemapb.DataType) error {
	return merr.WrapErrImportFailed(fmt.Sprintf("expected element type '%s' in array field, got type '%T' with value '%v'",
		eleType.String(), v, v))
}

func (p *colParser) wrapDimError(actualDim int, line int) error {
	field := p.fieldSchema
	return merr.WrapErrImportFailed(fmt.Sprintf("expected dim '%d' for field '%s' with type '%s', got dim '%d' in line %d",
		p.dim, field.GetName(), field.GetDataType().String(), actualDim, line))
}

func (p *colParser) getFieldID() int64 {
	return p.fieldSchema.FieldID
}

// parse 对一行数据的单个字段的解析
func (p *colParser) parse(records []string, line int) (any, error) {
	record := &records[p.index]
	return p.parseEntity(record, line)
}

// parseEntity 尝试将string类型转换成字段所需要的类型
func (p *colParser) parseEntity(data *string, line int) (any, error) {
	obj := *data
	switch p.fieldSchema.DataType {
	case schemapb.DataType_Bool:
		var value bool
		if obj == "0" {
			value = false
		} else if obj == "1" {
			value = true
		} else {
			return nil, p.wrapTypeError(obj, line)
		}
		return value, nil
	case schemapb.DataType_Int8:
		value, err := strconv.ParseInt(obj, 0, 8)
		if err != nil {
			return nil, p.wrapTypeWithError(obj, line, err)
		}
		return int8(value), nil
	case schemapb.DataType_Int16:
		value, err := strconv.ParseInt(obj, 0, 16)
		if err != nil {
			return nil, p.wrapTypeWithError(obj, line, err)
		}
		return int16(value), nil
	case schemapb.DataType_Int32:
		value, err := strconv.ParseInt(obj, 0, 32)
		if err != nil {
			return nil, p.wrapTypeWithError(obj, line, err)
		}
		return int32(value), nil
	case schemapb.DataType_Int64:
		value, err := strconv.ParseInt(obj, 0, 64)
		if err != nil {
			return nil, p.wrapTypeWithError(obj, line, err)
		}
		return value, nil
	case schemapb.DataType_Float:
		value, err := strconv.ParseFloat(obj, 32)
		if err != nil {
			return nil, p.wrapTypeWithError(obj, line, err)
		}
		return float32(value), nil
	case schemapb.DataType_Double:
		value, err := strconv.ParseFloat(obj, 64)
		if err != nil {
			return nil, p.wrapTypeWithError(obj, line, err)
		}
		return value, nil
	case schemapb.DataType_BinaryVector:
		arr := splitVec(data)
		if len(arr) != p.dim/8 {
			return nil, p.wrapDimError(len(arr)*8, line)
		}
		vec := make([]byte, len(arr))
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseUint(arr[i], 0, 8)
			if err != nil {
				return nil, p.wrapTypeWithError(arr[i], line, err)
			}
			vec[i] = byte(num)
		}
		return vec, nil
	case schemapb.DataType_FloatVector:
		arr := splitVec(data)
		if len(arr) != p.dim {
			return nil, p.wrapDimError(len(arr), line)
		}
		vec := make([]float32, len(arr))
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseFloat(arr[i], 32)
			if err != nil {
				return nil, p.wrapTypeWithError(arr[i], line, err)
			}
			vec[i] = float32(num)
		}
		return vec, nil
	case schemapb.DataType_Float16Vector, schemapb.DataType_BFloat16Vector:
		arr := splitVec(data)
		if len(arr) != p.dim*2 {
			return nil, p.wrapDimError(len(arr), line)
		}
		vec := make([]byte, len(arr))
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseUint(arr[i], 0, 8)
			if err != nil {
				return nil, p.wrapTypeWithError(arr[i], line, err)
			}
			vec[i] = byte(num)
		}
		return vec, nil
	case schemapb.DataType_SparseFloatVector:
		arr := splitVec(data)
		if len(arr)&8 != 0 {
			return nil, p.wrapDimError(len(arr), line)
		}
		vec := make([]byte, len(arr))
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseUint(arr[i], 0, 8)
			if err != nil {
				return nil, err
			}
			vec[i] = byte(num)
		}
		return vec, nil
	case schemapb.DataType_String, schemapb.DataType_VarChar:
		return obj, nil
	case schemapb.DataType_JSON:
		var dummy interface{}
		err := json.Unmarshal([]byte(obj), &dummy)
		if err != nil {
			return nil, err
		}
		return []byte(obj), nil
	case schemapb.DataType_Array:
		arr := splitVec(data)
		if len(arr) == 0 {
			return nil, p.wrapTypeError(obj, line)
		}
		scalarFieldData, err := p.arrayToFieldData(arr, p.fieldSchema.GetElementType(), line)
		if err != nil {
			return nil, err
		}
		return scalarFieldData, nil
	default:
		return nil, merr.WrapErrImportFailed(fmt.Sprintf("parse csv failed, unsupport data type: %s",
			p.fieldSchema.String()))
	}
}

// DataType_Array 参考json的实现
func (p *colParser) arrayToFieldData(arr []string, eleType schemapb.DataType, line int) (*schemapb.ScalarField, error) {
	switch eleType {
	case schemapb.DataType_Bool:
		values := make([]bool, 0)
		for i := 0; i < len(arr); i++ {
			var value bool
			if arr[i] == "0" || arr[i] == "false" || arr[i] == "FALSE" {
				value = false
			} else if arr[i] == "1" || arr[i] == "true" || arr[i] == "TRUE" {
				value = true
			} else {
				return nil, p.wrapTypeError(arr[i], line)
			}
			values = append(values, value)
		}
		return &schemapb.ScalarField{
			Data: &schemapb.ScalarField_BoolData{
				BoolData: &schemapb.BoolArray{
					Data: values,
				},
			},
		}, nil
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
		values := make([]int32, 0)
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseInt(arr[i], 0, 32)
			if err != nil {
				return nil, p.wrapTypeWithError(arr[i], line, err)
			}
			values = append(values, int32(num))
		}
		return &schemapb.ScalarField{
			Data: &schemapb.ScalarField_IntData{
				IntData: &schemapb.IntArray{
					Data: values,
				},
			},
		}, nil
	case schemapb.DataType_Int64:
		values := make([]int64, 0)
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseInt(arr[i], 0, 64)
			if err != nil {
				return nil, p.wrapTypeWithError(arr[i], line, err)
			}
			values = append(values, num)
		}
		return &schemapb.ScalarField{
			Data: &schemapb.ScalarField_LongData{
				LongData: &schemapb.LongArray{
					Data: values,
				},
			},
		}, nil
	case schemapb.DataType_Float:
		values := make([]float32, 0)
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseFloat(arr[i], 32)
			if err != nil {
				return nil, p.wrapTypeWithError(arr[i], line, err)
			}
			values = append(values, float32(num))
		}
		return &schemapb.ScalarField{
			Data: &schemapb.ScalarField_FloatData{
				FloatData: &schemapb.FloatArray{
					Data: values,
				},
			},
		}, nil
	case schemapb.DataType_Double:
		values := make([]float64, 0)
		for i := 0; i < len(arr); i++ {
			num, err := strconv.ParseFloat(arr[i], 64)
			if err != nil {
				return nil, p.wrapTypeWithError(arr[i], line, err)
			}
			values = append(values, num)
		}
		return &schemapb.ScalarField{
			Data: &schemapb.ScalarField_DoubleData{
				DoubleData: &schemapb.DoubleArray{
					Data: values,
				},
			},
		}, nil
	case schemapb.DataType_VarChar, schemapb.DataType_String:
		values := make([]string, 0)
		for i := 0; i < len(arr); i++ {
			values = append(values, arr[i])
		}
		return &schemapb.ScalarField{
			Data: &schemapb.ScalarField_StringData{
				StringData: &schemapb.StringArray{
					Data: values,
				},
			},
		}, nil
	default:
		return nil, fmt.Errorf("unsupported array data type '%s'", eleType.String())
	}
}

type dynamicFieldParser struct {
	index       int
	fieldSchema *schemapb.FieldSchema
	name2index  map[string]int
}

func (p *dynamicFieldParser) getFieldID() int64 {
	return p.fieldSchema.FieldID
}

func (p *dynamicFieldParser) parse(records []string, line int) (any, error) {
	if p.index >= len(records) {
		return []byte("{}"), nil
	}
	var mp map[string]interface{}
	if p.index != -1 {
		record := &records[p.index]
		err := json.Unmarshal([]byte(*record), &mp)
		if err != nil {
			return nil, merr.WrapErrImportFailed(fmt.Sprintf("invalid JSON format, each row should be a key-value map, error: %v ,in line %d", err, line))
		}
	}
	if mp == nil {
		mp = make(map[string]interface{})
	}
	for name, index := range p.name2index {
		mp[name] = p.convert(&records[index])
	}
	if len(mp) == 0 {
		return []byte("{}"), nil
	}
	bs, err := json.Marshal(mp)
	if err != nil {
		return nil, err
	}
	return bs, nil
}

// convert 因为dynamicField的底层实现是json，
// 所以convert会尝试将string转换成bool,int,float以及vector
// 如果都没有成功，那就当成string类型
func (p *dynamicFieldParser) convert(data *string) any {
	obj := *data
	if len(obj) == 0 {
		return ""
	}
	if len(obj) == 1 {
		if num, ok := p.isInteger(data); ok {
			return num
		}
		return obj
	}
	if obj[0] == obj[len(obj)-1] && obj[0] == '"' {
		return obj
	}
	if b, ok := p.isBool(data); ok {
		return b
	}
	if num, ok := p.isInteger(data); ok {
		return num
	}
	if num, ok := p.isFloater(data); ok {
		return num
	}
	if vec, ok := p.isVec(data); ok {
		return vec
	}
	return obj
}

func (p *dynamicFieldParser) isBool(data *string) (bool, bool) {
	obj := *data
	if obj == "TRUE" || obj == "true" {
		return true, true
	}
	if obj == "FALSE" || obj == "false" {
		return false, true
	}
	return false, false
}

func (p *dynamicFieldParser) isInteger(data *string) (int32, bool) {
	num, err := strconv.ParseInt(*data, 0, 32)
	return int32(num), err == nil
}

func (p *dynamicFieldParser) isFloater(data *string) (float32, bool) {
	num, err := strconv.ParseFloat(*data, 32)
	return float32(num), err == nil
}

func (p *dynamicFieldParser) isVec(data *string) (any, bool) {
	obj := *data
	if len(obj) < 2 {
		return "", false
	}
	if obj[0] != '[' || obj[len(obj)-1] != ']' {
		return "", false
	}
	d := make([]any, 0)
	vec := splitVec(data)
	if len(vec) == 0 {
		return "[]", true
	}
	for _, value := range vec {
		d = append(d, p.convert(&value))
	}
	return d, true
}

// 以逗号作为分割符来切割record
func splitVec(data *string) []string {
	obj := *data
	if len(obj) > 1 {
		if obj[0] == '[' && obj[len(obj)-1] == ']' {
			obj = obj[1 : len(obj)-1]
		}
	}
	return strings.Split(obj, ",")
}
