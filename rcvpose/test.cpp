// test.cpp : Defines the entry point for the application.
//

#include "rcvpose.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

Options testing_options() {
    Options opts;
    opts.dname = "bw";
    //opts.root_dataset = "C:\\RCVPose\\dataset";
    opts.root_dataset = "C:\\RCVPose\\Datasets\\bwpnew1";
    //opts.root_dataset = "C:\\RCVPose\\simData\\BinGen\\DATASET";
    //opts.model_dir = "C:\\RCVPose\\win_cpp_rcvpose\\rcvpose\\trained_model";
    opts.model_dir = "C:\\RCVPose\\models\\nathan_6_24_24_bwp1";
    opts.resume_train = false;
    opts.optim = "adam";
    opts.frontend = "accumulator";
    opts.batch_size = 8;
    //opts.class_name = "ape";
    opts.class_name = "bwp2";
    opts.initial_lr = 0.001;
    opts.reduce_on_plateau = true;
    opts.patience = 10;
    opts.demo_mode = true;
    opts.verbose = true;
    opts.test_occ = false;
    opts.mask_threshold = 0.8;
    opts.epsilon = 0.01;
    opts.use_gt = false;
    return opts;
}


int main(int argc, char* args[])
{
    bool train = false;
    bool validate = false;
    bool estimate = false;

    if(argc > 1){
        if (strcmp(args[1], "train") == 0){
            train = true;
        }
        else if (strcmp(args[1], "validate") == 0){
            validate = true;
        }
        else if (strcmp(args[1], "estimate") == 0){
            estimate = true;
        }
        else {
            cout << "Usage: " << args[0] << " <train/validate/estimate>" << endl;
            return 0;
        }
    } else {
        cout << "Usage: " << args[0] << " <train/validate/estimate>" << endl;
        cout << "Defaulting to validating" << endl;
        validate = true;
    }

    Options opts;
    if ((argc > 2) &&(argc < 15)) {
        try {
            opts.dname = args[2];

            opts.root_dataset = args[3];

            opts.model_dir = args[4];

            if (args[5] == "true") {
                opts.resume_train = true;
            }
            else {
                opts.resume_train = false;
            }
            opts.optim = args[6];

            opts.batch_size = stoi(args[7]);

            opts.class_name = args[8];

            opts.initial_lr = stod(args[9]);
            if (args[10] == "true") {
                opts.reduce_on_plateau = true;
            }
            else {
                opts.reduce_on_plateau = false;
            }
            opts.patience = stoi(args[12]);
            if (args[12] == "true") {
                opts.demo_mode = true;
            }
            else {
                opts.demo_mode = false;
            }
            if (args[13] == "true") {
                opts.verbose = true;
            }
            else {
                opts.verbose = false;
            }
            if (args[14] == "true") {
                opts.test_occ = true;
            }
            else {
                opts.test_occ = false;
            }
        }
        catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
            cout << "Usage: " << args[0] << " <dataset_name(lm)> <dataset_root> <model_directory> <resume_train(true/false)> <optim(adam/sgd)> <batch_size(int)> <class_name(string)> <initial_lr(double)> <reduce_on_plateau(true/false)> <patience(int)> <demo_mode(true/false)> <verbose(true/false)> <test_occ(true/false)(hasn't been implemented yet)>" << endl;
            cout << "Defaulting to testing options" << endl;
            opts = testing_options();
        }
    }
    else {
        cout << "Using Default Testing Options" <<  endl;
        opts = testing_options();
    }

    // Ensure model is loaded for validate or estimate option
    if (validate || estimate) opts.resume_train = true;

    RCVpose rcv(opts);
    //Trains the model with the given parameters, if resume if true, will resume training from previous saved state
    if (train){
        //opts.resume_train = true;
        rcv.train();

    }


    // Runs through the entire test dataset and prints the ADD before and after ICP as well as time taken
    if (validate)
        rcv.validate();

    // Estimates the pose of a single input RGBD image and prints the estimated pose as well as time taken
    if(estimate){
        for (int i = 1; i < 100; i++) {
            cout << "Estimating..." << endl;
            string img_num_str_offset = to_string(i + 4500);

            string padded_img_num_offset = string(6 - img_num_str_offset.length(), '0') + img_num_str_offset;

            string img_path = opts.root_dataset + "/sim/JPEGImages/" + padded_img_num_offset + ".jpg";
            string depth_path = opts.root_dataset + "/sim/data/depth" + img_num_str_offset + ".png";

            cout << depth_path << endl;

            rcv.estimate_pose(img_path, depth_path);
        }
        return 0;
    }

    return 0;
}
