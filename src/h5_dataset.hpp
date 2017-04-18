#include <v8.h>
#include <uv.h>
#include <node.h>
#include <node_buffer.h>

#include <algorithm>
#include <cstring>
#include <vector>
#include <map>
#include <functional>
#include <memory>
// #include <iostream>

#include "file.h"
#include "group.h"
#include "int64.hpp"
#include "H5LTpublic.h"

namespace NodeHDF5 {

    v8::Local<v8::Array> CreateArraysForNextRank2(std::vector<hsize_t> max_dims, uint32_t index, std::vector<uint32_t> strideTable, std::vector<uint64_t> iterLoc, double * elements) {
        v8::Local<v8::Array> array = v8::Array::New(v8::Isolate::GetCurrent(), max_dims[index]);

        if (index == 0) {
            uint32_t base_offset = 0;

            for (uint32_t index = 0; index < max_dims.size(); index++) {
                base_offset += strideTable[index] * iterLoc[index];
            }

            for(unsigned int elementIndex = 0; elementIndex < max_dims[0]; elementIndex++){
                uint32_t offset = base_offset + (elementIndex * strideTable[0]);
                array->Set(elementIndex, v8::Number::New(v8::Isolate::GetCurrent(), elements[offset]));
            }

            return array;
        }

        for(uint32_t elementIndex = 0; elementIndex < max_dims[index]; elementIndex++) {
            array->Set(elementIndex, CreateArraysForNextRank2(max_dims, index - 1, strideTable, iterLoc, elements));
            iterLoc[index]++;
        }

        iterLoc[index] = 0;

        return array;
    }

    class H5D {
    protected:

    public:
        static void Initialize (Handle<Object> target) {
            target->Set(String::NewFromUtf8(v8::Isolate::GetCurrent(), "openDataset"), FunctionTemplate::New(v8::Isolate::GetCurrent(), H5D::open_dataset)->GetFunction());
            target->Set(String::NewFromUtf8(v8::Isolate::GetCurrent(), "closeDataset"), FunctionTemplate::New(v8::Isolate::GetCurrent(), H5D::close_dataset)->GetFunction());
            target->Set(String::NewFromUtf8(v8::Isolate::GetCurrent(), "readDatasetHyperSlab"), FunctionTemplate::New(v8::Isolate::GetCurrent(), H5D::read_dataset_hyper_slab)->GetFunction());
        }

        static void open_dataset (const v8::FunctionCallbackInfo<Value>& args)
        {
            // Fail if arguments are not correct
            if(args.Length() == 3 && (!args[0]->IsObject() || !args[1]->IsString() || !args[2]->IsObject())){
                v8::Isolate::GetCurrent()->ThrowException(v8::Exception::SyntaxError(String::NewFromUtf8(v8::Isolate::GetCurrent(), "expected id, name, options")));
                args.GetReturnValue().SetUndefined();
                return;

            }
            else if (args.Length() ==2 && (!args[0]->IsObject() || !args[1]->IsString())) {

                v8::Isolate::GetCurrent()->ThrowException(v8::Exception::SyntaxError(String::NewFromUtf8(v8::Isolate::GetCurrent(), "expected id, name")));
                args.GetReturnValue().SetUndefined();
                return;
            }

            String::Utf8Value dataset_name(args[1]->ToString());
            int rank;
            Int64* idWrap = ObjectWrap::Unwrap<Int64>(args[0]->ToObject());
            herr_t err = H5LTget_dataset_ndims(idWrap->Value(), *dataset_name, &rank);

            if(err < 0)
            {
                v8::Isolate::GetCurrent()->ThrowException(v8::Exception::SyntaxError(String::NewFromUtf8(v8::Isolate::GetCurrent(), "failed to find dataset rank")));
                args.GetReturnValue().SetUndefined();
                return;
            }

            size_t bufSize = 0;
            H5T_class_t class_id;
            std::vector<hsize_t> values_dim(rank);
            err = H5LTget_dataset_info(idWrap->Value(), *dataset_name, values_dim.data(), &class_id, &bufSize);

            v8::Local<v8::Array> array = v8::Array::New(v8::Isolate::GetCurrent(), rank);

            for(int dimIndex = 0; dimIndex < rank; dimIndex++){
                array->Set(dimIndex, v8::Number::New(v8::Isolate::GetCurrent(), values_dim[dimIndex]));
            }

            if(err < 0)
            {
                v8::Isolate::GetCurrent()->ThrowException(v8::Exception::SyntaxError(String::NewFromUtf8(v8::Isolate::GetCurrent(), "failed to find dataset info")));
                args.GetReturnValue().SetUndefined();
                return;
            }

            std::vector<hsize_t> count;

            if(args.Length() == 3){
                Local<Array> names=args[2]->ToObject()->GetOwnPropertyNames();
                for(uint32_t index=0;index<names->Length();index++){
                    String::Utf8Value _name (names->Get(index));
                    std::string name(*_name);
                    if(name.compare("count")==0){
                        Local<Object> counts=args[2]->ToObject()->Get(names->Get(index))->ToObject();
                        for(unsigned int arrayIndex=0;arrayIndex<counts->Get(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "length"))->ToObject()->Uint32Value();arrayIndex++){
                            count.push_back(counts->Get(arrayIndex)->Uint32Value());
                        }
                    }
                }
            }


            hid_t did = H5Dopen(idWrap->Value(), *dataset_name, H5P_DEFAULT);
            hid_t type_id;

            if(class_id==H5T_FLOAT && bufSize==8)
            {
                type_id=H5T_NATIVE_DOUBLE;
            }
            else if(class_id==H5T_FLOAT && bufSize==4)
            {
                type_id=H5T_NATIVE_FLOAT;
            }
            else if(class_id==H5T_INTEGER && bufSize==4)
            {
                if(H5Tget_sign(H5Dget_type(did))==H5T_SGN_2)
                {
                    type_id=H5T_NATIVE_INT;
                }
                else
                {
                    type_id=H5T_NATIVE_UINT;
                }
            }
            else if(class_id==H5T_INTEGER && bufSize==2)
            {
                 if(H5Tget_sign(H5Dget_type(did))==H5T_SGN_2)
                 {
                     type_id=H5T_NATIVE_SHORT;
                 }
                 else
                 {
                     type_id=H5T_NATIVE_USHORT;
                 }
            }
            else if(class_id==H5T_INTEGER && bufSize==1)
            {
                 if(H5Tget_sign(H5Dget_type(did))==H5T_SGN_2)
                 {
                     type_id=H5T_NATIVE_INT8;
                 }
                 else
                 {
                     type_id=H5T_NATIVE_UINT8;
                 }
            }
            else
            {
                v8::Isolate::GetCurrent()->ThrowException(v8::Exception::SyntaxError(String::NewFromUtf8(v8::Isolate::GetCurrent(), "unsupported data type")));
                args.GetReturnValue().SetUndefined();
                return;
            }

            hid_t dataspace_id = H5S_ALL;
            hid_t memspace_id = H5S_ALL;

            std::vector<hsize_t> max_dims(rank);

            for(int rankIndex=0; rankIndex < rank; rankIndex++){
                max_dims[rankIndex] = H5S_UNLIMITED;
            }

            memspace_id = H5Screate_simple (rank, count.data(), max_dims.data());
            dataspace_id = H5Dget_space (did);

            v8::Local<v8::Object> object = v8::Object::New(v8::Isolate::GetCurrent());

            Local<Object> datasetIdInstance = Int64::Instantiate(args.This(), did);
            Int64 *  datasetWrap = ObjectWrap::Unwrap<Int64>(datasetIdInstance);
            datasetWrap->value = did;

            object->Set(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "dataset"), datasetIdInstance);

            Local<Object> memspaceIdInstance = Int64::Instantiate(args.This(), memspace_id);
            Int64 * memspaceWrap = ObjectWrap::Unwrap<Int64>(memspaceIdInstance);
            memspaceWrap->value = memspace_id;

            object->Set(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "memspace"), memspaceIdInstance);

            Local<Object> dataspaceIdInstance = Int64::Instantiate(args.This(), dataspace_id);
            Int64 * dataspaceWrap = ObjectWrap::Unwrap<Int64>(dataspaceIdInstance);
            dataspaceWrap->value = dataspace_id;

            object->Set(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "dataspace"), dataspaceIdInstance);

            object->Set(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "rank"), Number::New(v8::Isolate::GetCurrent(), rank));

            object->Set(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "dims"), array);

            args.GetReturnValue().Set(object);
        }

        static void read_dataset_hyper_slab (const v8::FunctionCallbackInfo<Value>& args)
        {
            Int64 * memspaceWrap = ObjectWrap::Unwrap<Int64>(args[0]->ToObject());
            hid_t memspace_id = memspaceWrap->Value();

            Int64 * dataspaceWrap = ObjectWrap::Unwrap<Int64>(args[1]->ToObject());
            hid_t dataspace_id = dataspaceWrap->Value();

            Int64 * datasetWrap = ObjectWrap::Unwrap<Int64>(args[2]->ToObject());
            hid_t did = datasetWrap->Value();

            int rank = args[3]->Uint32Value();

            std::vector<hsize_t> start;
            std::vector<hsize_t> stride;
            std::vector<hsize_t> count;

            Local<Array> names=args[4]->ToObject()->GetOwnPropertyNames();
            for(uint32_t index=0;index<names->Length();index++){
                String::Utf8Value _name (names->Get(index));
                std::string name(*_name);
                if(name.compare("start")==0){
                    Local<Object> starts=args[4]->ToObject()->Get(names->Get(index))->ToObject();
                    for(unsigned int arrayIndex=0;arrayIndex<starts->Get(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "length"))->ToObject()->Uint32Value();arrayIndex++){
                        start.push_back(starts->Get(arrayIndex)->Uint32Value());
                    }
                }
                else if(name.compare("stride")==0){
                    Local<Object> strides=args[4]->ToObject()->Get(names->Get(index))->ToObject();
                    for(unsigned int arrayIndex=0;arrayIndex<strides->Get(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "length"))->ToObject()->Uint32Value();arrayIndex++){
                        stride.push_back(strides->Get(arrayIndex)->Uint32Value());
                    }
                }
                else if(name.compare("count")==0){
                    Local<Object> counts=args[4]->ToObject()->Get(names->Get(index))->ToObject();
                    for(unsigned int arrayIndex=0;arrayIndex<counts->Get(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "length"))->ToObject()->Uint32Value();arrayIndex++){
                        count.push_back(counts->Get(arrayIndex)->Uint32Value());
                    }
                }
            }

            herr_t  err = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, start.data(),
                                               stride.data(), count.data(), NULL);
            if(err < 0)
            {
                v8::Isolate::GetCurrent()->ThrowException(v8::Exception::SyntaxError(String::NewFromUtf8(v8::Isolate::GetCurrent(), "failed to select hyperslab")));
                args.GetReturnValue().SetUndefined();
                return;
            }

            hsize_t theSize = 1;

            std::vector<hsize_t> max_dims(rank);

            for(int rankIndex=0; rankIndex < rank; rankIndex++){
                max_dims[rankIndex] = count[rankIndex];
                theSize *= count[rankIndex];
            }

            std::unique_ptr<double> buf(new double[theSize]);

            hid_t t = H5Dget_type(did);

            err = H5Dread(did, t, memspace_id, dataspace_id, H5P_DEFAULT, buf.get());

            H5Tclose(t);

            if(err<0)
            {
                v8::Isolate::GetCurrent()->ThrowException(v8::Exception::SyntaxError(String::NewFromUtf8(v8::Isolate::GetCurrent(), "failed to read dataset")));
                args.GetReturnValue().SetUndefined();
                return;
            }

            //
            std::vector<uint64_t> iterLoc(rank, 0);

            std::vector<uint32_t> strideTable(rank, 1);

            for (uint32_t index = 0; index < (max_dims.size() - 1); index++) {
                uint32_t offset = 1;

                for (uint32_t jndex = index + 1; jndex < max_dims.size(); jndex++) {
                    offset *= max_dims[jndex];
                }

                strideTable[index] = offset;
            }

            v8::Local<v8::Array> array = CreateArraysForNextRank2(max_dims, rank - 1, strideTable, iterLoc, buf.get());

            v8::Local<v8::Object> object=v8::Object::New(v8::Isolate::GetCurrent());

            object->Set(v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), "data"), array);

            args.GetReturnValue().Set(object);
        }

        static void close_dataset (const v8::FunctionCallbackInfo<Value>& args)
        {
            Int64 * memspaceWrap = ObjectWrap::Unwrap<Int64>(args[0]->ToObject());
            H5Sclose(memspaceWrap->Value());

            Int64 * dataspaceWrap = ObjectWrap::Unwrap<Int64>(args[1]->ToObject());
            H5Sclose(dataspaceWrap->Value());

            Int64 * datasetWrap = ObjectWrap::Unwrap<Int64>(args[2]->ToObject());
            H5Dclose(datasetWrap->Value());
        }
    };
}
