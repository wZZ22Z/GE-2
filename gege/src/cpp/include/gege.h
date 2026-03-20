#pragma once

#include "configuration/config.h"
#include "data/dataloader.h"
#include "nn/model.h"
#include "storage/graph_storage.h"

void encode_and_export(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<GegeConfig> gege_config);

std::tuple<shared_ptr<Model>, shared_ptr<GraphModelStorage>, shared_ptr<DataLoader> > gege_init(shared_ptr<GegeConfig> gege_config, bool train);

void gege_train(shared_ptr<GegeConfig> gege_config);

void gege_eval(shared_ptr<GegeConfig> gege_config);

void gege(int argc, char *argv[]);

int main(int argc, char *argv[]);
