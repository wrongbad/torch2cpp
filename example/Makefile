INCLUDES := $(shell python -m torch2cpp.includes)

build/model.js : build/model.cpp
	em++ -Os build/model.cpp -I$(INCLUDES) \
		-o build/model.js -s MODULARIZE=1 -s EXPORT_NAME=load_model \
		-s EXPORTED_FUNCTIONS=_model_step,_model_reset,_model_encode,_model_decode

build/chat_cli : chat_cli.cpp build/model.cpp 
	c++ -std=c++17 -Os -march=native -ffast-math \
		build/model.cpp chat_cli.cpp -I$(INCLUDES) -o build/chat_cli

.PHONY: model.js
model.js: build/model.js

.PHONY: chat_cli
chat_cli: build/chat_cli