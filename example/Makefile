INCLUDES := $(shell python -m torch2cpp.includes)

build/model.js : build/model.cpp
	em++ $^ -o $@ -I$(INCLUDES) \
		-Os -s MODULARIZE=1 -s EXPORT_NAME=load_model \
		-s EXPORTED_FUNCTIONS=_model_step,_model_reset,_model_encode,_model_decode,_malloc

build/chat_cli : build/model.cpp chat_cli.cpp
	c++ $^ -o $@ -I$(INCLUDES) \
		-Os -std=c++17 -march=native -ffast-math

.PHONY: model.js
model.js: build/model.js

.PHONY: chat_cli
chat_cli: build/chat_cli