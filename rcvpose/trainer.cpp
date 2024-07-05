#include "trainer.h"

using namespace std;


Trainer::Trainer(const Options& opts, DenseFCNResNet152& model) 
{
    cout << string(100, '=') << endl;
    cout << string (35, ' ') << "Initializing Trainer" << endl << endl;

    bool use_cuda = torch::cuda::is_available();
    device_type = use_cuda ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);

    model->to(device);

    if (opts.verbose) {
        cout << "Using " << (use_cuda ? "CUDA" : "CPU") << endl;
        cout << "Setting up model" << endl;
    }
    out = opts.model_dir;

    if (!opts.resume_train) {
        if (opts.verbose) {
            cout << "Setting up optimizer" << endl;
        }
        try {
            if (opts.optim == "adam") {
                optim  = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(opts.initial_lr));

            }
            else if (opts.optim == "sgd") {
                optim = new torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(opts.initial_lr));
            }
            else {
                cout << "Error: Invalid optimizer" << endl;
                return;
            }
        }
        catch (const torch::Error& e) {
            cout << "Error: " << e.msg() << endl;
            return;
        }
        int count = 0;
        for (auto& params : optim->param_groups()) {
            if (opts.verbose) {
                cout << "Param Group " << count << " with LR value: " << static_cast<torch::optim::AdamOptions &>(params.options()).lr();//params.options().get_lr() << endl;
            }
            count++;
        }

        current_lr.clear();
        current_lr.push_back(opts.initial_lr);

        epoch = 0;
        starting_epoch = 0;
        
    } 
    else {
        try {
            cout << "Loading model from checkpoint" << endl;
            CheckpointLoader loader(out, true);
            epoch = loader.getEpoch();
            starting_epoch = epoch;
            if (opts.verbose) {
                cout << "Epoch: " << epoch << endl;
            }
            optim = loader.getOptimizer();
            optim->parameters() = model->parameters();
            current_lr = loader.getLrList();
            if (opts.verbose) {
                cout << "Optimizer loaded" << endl;
            }
            int count = 0;
            for (auto& params : optim->param_groups()) {
		static_cast<torch::optim::AdamOptions &>(params.options()).lr(current_lr[0]);
		//params.options().set_lr(current_lr[0]);
                if (opts.verbose) {
                    cout << "Param Group " << count << " with LR value: " << static_cast<torch::optim::AdamOptions &>(params.options()).lr() <<endl; //params.options().get_lr() << endl;
                }
                count++;
            }

            current_lr.clear();
            current_lr.push_back(opts.initial_lr);

            best_acc_mean = loader.getBestAccuracy();

            float prev_loss = loader.getLoss();
            if (opts.verbose) {
                cout << "Best Accuracy: " << best_acc_mean << endl;

                cout << "Previous Loss: " << prev_loss << endl;
            }

        } 
        catch (const torch::Error& e) {
            cout << "Cannot Resume Training" << endl;
			cout << "Error: " << e.msg() << endl;
            return;
		}
    }

    
    if (opts.verbose) {
        cout << "Setting up loss function" << endl;
    }

    try {
        loss_radial = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kSum));
        loss_radial->to(device);
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }
    try {
        loss_sem = torch::nn::BCEWithLogitsLoss(torch::nn::BCEWithLogitsLossOptions().weight(torch::tensor(38)));//torch::nn::L1Loss();
        loss_sem->to(device);
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }

    iteration = 0;
    iteration_val = 0;
    max_iteration = opts.cfg.at("max_iteration")[0];
    best_acc_mean = std::numeric_limits<double>::infinity();


    std::filesystem::path outPath(out);
    if (!std::filesystem::is_directory(outPath)) {
        if (std::filesystem::create_directories(outPath)) {
            std::cout << "Output directory created" << std::endl;
        }
        else {
            std::cout << "Failed to create output directory" << std::endl;
        }
    }
    else {
        std::cout << "Output directory already exists" << std::endl;
    }


    cout << "Model Path: " << out << endl;

    cout << "Trainer Initialized" << endl;

    
}

void Trainer::train(Options& opts, DenseFCNResNet152& model)
{
    cout << string(100, '=') << endl; 
    cout << string(24, ' ') << "Begining Training Initialization" << endl << endl;

    // Start log file
    string logFilePath = opts.model_dir + "/training_log.txt";
    ofstream outFile(logFilePath, ios_base::trunc);
    time_t now = time(nullptr);
    tm* localTime = localtime(&now);
    outFile << string(50, '#') << endl << "Training Started - " << put_time(localTime, "%Y-%m-%d %H:%M:%S") << endl << string(50, '#') << endl;
    outFile.flush();

    torch::Device device(device_type);
    if (opts.verbose) {
        cout << "Setting up dataset loader" << endl;
    }
    
    //Note that val and train are swapped here, this is done to improve the quality of the trained model
    auto train_dataset = RData(opts.root_dataset, opts.dname, "val", opts.class_name);
    torch::optional<size_t> train_size = train_dataset.size();

    auto val_dataset = RData(opts.root_dataset, opts.dname, "train", opts.class_name);
    torch::optional<size_t> val_size = val_dataset.size();


 
    if (train_size.value() == 0 || !train_size.has_value()) {
        cout << "Error: Could not get size of training dataset" << endl;
        return;
    }
    if (val_size.value() == 0 || !val_size.has_value()) {
        cout << "Error: Could not get size of validation dataset" << endl;
        return;
    }

    cout << "Train Data Set Size : " << train_size.value() << endl;
    cout << "Val Data Set Size : " << val_size.value() << endl;

    max_epoch = static_cast<int>(std::ceil(1.0 * max_iteration / train_size.value()));


    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
    );

    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset),
        torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
    );

    cout << "Max Epochs : " << max_epoch << endl;
    
    int session_iterations = 0;

    auto total_train_start = std::chrono::steady_clock::now();
    while (epoch < max_epoch) {
        auto epoch_start_time = std::chrono::steady_clock::now();

        //Log start of new epoch
        outFile << string(50, '-') << endl << "### Start of Epoch " << epoch << " ###" << endl;
        outFile.flush();

        if (opts.verbose) {
            cout << string(100, '-') << endl;
            cout << string(43, ' ') << "Epoch " << epoch << endl;
            cout << "Training Epoch" << endl;
        }

        // ========================================================================================== \\
        // ====================================== Training ========================================== \\
        
        
        int count = 0;
        model->train();


        auto train_start = std::chrono::steady_clock::now();
        int train_pix_gath = 0;


        for (const auto& batch : *train_loader) {
            //if (opts.verbose) {
            //    printProgressBar(count, train_size.value(), 75);
            //}

            iteration = batch.size() + iteration;

            std::vector<std::vector<torch::Tensor>> batches(5);

            for (const auto& example : batch) {
                batches[0].push_back(example.data());
                batches[1].push_back(example.rad_1());
                batches[2].push_back(example.rad_2());
                batches[3].push_back(example.rad_3());
                batches[4].push_back(example.sem_target());
            }

            auto data = torch::stack(batches[0], 0).to(device);
            auto rad_1 = torch::stack(batches[1], 0).to(device);
            auto rad_2 = torch::stack(batches[2], 0).to(device);
            auto rad_3 = torch::stack(batches[3], 0).to(device);
            auto sem_target = torch::stack(batches[4], 0).to(device);
            auto sem_target_permuted = sem_target.permute({ 1, 0, 2, 3 });

            optim->zero_grad();

            if ((session_iterations == 0) && (opts.verbose) && (count == 0) && (device == torch::kCUDA)) {
                cout << "\r" << string(100, ' ') << "\r";
                cout << "Training GPU memory usage" << endl;
                printGPUmem();
            }

            torch::Tensor scores = model->forward(data);

            auto score_rad_1 = scores.index({ torch::indexing::Slice(), 0 }).unsqueeze(1);
            auto score_rad_2 = scores.index({ torch::indexing::Slice(), 1 }).unsqueeze(1);
            auto score_rad_3 = scores.index({ torch::indexing::Slice(), 2 }).unsqueeze(1);
            auto score_sem = scores.index({ torch::indexing::Slice(), 3 }).unsqueeze(1);

            score_rad_1 = (score_rad_1.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 }));
            score_rad_2 = (score_rad_2.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 }));
            score_rad_3 = (score_rad_3.permute({ 1, 0, 2, 3 }) * sem_target.permute({ 1, 0, 2, 3 }));

            score_rad_1 = score_rad_1.permute({ 1, 0, 2, 3 });
            score_rad_2 = score_rad_2.permute({ 1, 0, 2, 3 });
            score_rad_3 = score_rad_3.permute({ 1, 0, 2, 3 });
            
            //score_sem = torch::sigmoid(score_sem);	

            torch::Tensor loss_s = loss_sem(score_sem, sem_target);
            torch::Tensor loss_r = compute_r_loss(score_rad_1, rad_1);
            loss_r += compute_r_loss(score_rad_2, rad_2);
            loss_r += compute_r_loss(score_rad_3, rad_3);

            //score_sem = torch::sigmoid(score_sem); 
            
	        auto sem_target_ = torch::stack(batches[4], 0);
            auto gt1Tensor = torch::stack(batches[1], 0);
            auto gt2Tensor = torch::stack(batches[2], 0);
            auto gt3Tensor = torch::stack(batches[3], 0);

            torch::Tensor score_sem_ = score_sem.to(torch::kCPU);

            //std::cout<<score_sem_.sizes()<<std::endl;
            cv::Mat sem_cv = torch_tensor_to_cv_mat(sem_target_[0][0]);
            cv::Mat sem_cv_ = torch_tensor_to_cv_mat(score_sem_[0][0]);
            //cv::Mat rad_cv1 = torch_tensor_to_cv_mat(score_rad_1.to(torch::kCPU)[0][0]);
            cv::Mat rad_cv1 = torch_tensor_to_cv_mat(gt1Tensor[0][0]);
            cv::Mat rad_cv2 = torch_tensor_to_cv_mat(gt2Tensor[0][0]);
            cv::Mat rad_cv3 = torch_tensor_to_cv_mat(gt3Tensor[0][0]);
            
	        cv::transpose(sem_cv, sem_cv);
            cv::transpose(sem_cv_, sem_cv_);

	        double min;
            double max;
            cv::Point min_loc;
            cv::Point max_loc;
            /*minMaxLoc(sem_cv, &min, &max, &min_loc, &max_loc);
            std::cout << "SEM Max: " << max << " SEM Min: " << min << std::endl;*/
            
            /*minMaxLoc(sem_cv_, &min, &max, &min_loc, &max_loc);
            std::cout << "SEM Max_: " << max << " SEM Min_: " << min << std::endl;*/
            
            /*minMaxLoc(rad_cv1, &min, &max, &min_loc, &max_loc);
            std::cout << "RAD1 Max: " << max << " RAD1 Min: " << min << std::endl;*/
            
            /*minMaxLoc(rad_cv2, &min, &max, &min_loc, &max_loc);
            std::cout << "RAD2 Max: " << max << " RAD2 Min: " << min << std::endl;*/
            
            /*minMaxLoc(rad_cv3, &min, &max, &min_loc, &max_loc);
            std::cout << "RAD3 Max: " << max << " RAD3 Min: " << min << std::endl;*/

	        //cv::normalize(sem_cv_, sem_cv_, 0, 1, cv::NORM_MINMAX);
            
	        cv::Mat thresholded, thresholded_;
            cv::threshold(sem_cv, thresholded, 0.5, 1, cv::THRESH_BINARY);
            thresholded.convertTo(sem_cv, sem_cv.type());
            thresholded.release();
            
            cv::threshold(sem_cv_, thresholded_, 0.5, 1, cv::THRESH_BINARY);
            thresholded_.convertTo(sem_cv_, sem_cv_.type());
            thresholded_.release();
            
            vector<Vertex> pixel_coor;
            for (int i = 0; i < sem_cv.rows; i++) {
                for (int j = 0; j < sem_cv.cols; j++) {
                    if (sem_cv_.at<float>(i, j) == 1) {
                    	 //std::cout<<sem_cv.at<float>(i, j)<<std::endl;
                        Vertex v;
                        v.x = static_cast<double>(i);
                        v.y = static_cast<double>(j);
                        v.z = 1;
                        pixel_coor.push_back(v);
                    }
                }
            }

            cout << "Number of pixels gathered: " << pixel_coor.size() << endl;
            train_pix_gath += pixel_coor.size();
            
            cv::imwrite(opts.model_dir + "/asdf.png", sem_cv*255);
            cv::imwrite(opts.model_dir + "/asdf_.png", sem_cv_*255);
        	
            // Look into different weigthings for loss
            torch::Tensor loss = loss_r + loss_s;
            
            cout << "Radial Loss: " << loss_r.item<float>() << " Semantic Loss: " << loss_s.item<float>() << " Total Loss: " << loss.item<float>() << endl;

            loss.backward();

            optim->step();
		
            auto np_loss = loss.detach().cpu().numpy_T();

            if (np_loss.numel() == 0)
                std::runtime_error("Loss is empty");
            count = batch.size() + count;

            cout /*<< string(10, '/')*/ << endl;
        }

        // log pixels gathered
        //outFile << "Avg Pixels Gathered Train: " << train_size.value() << " -- " << (train_pix_gath / train_size.value()) << endl << endl;
        //outFile.flush();

        auto train_end = std::chrono::steady_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);
        if (opts.verbose) {
            cout << "\r" << string(80, ' ') << "\r\r";
            cout << "Training Time: " << train_duration.count() << " s" << endl;
        }
        

        // ========================================================================================== \\
        //                                  Validation Epoch 									       \\

        //if ((epoch % 3 == 0) && (epoch != 0)) {
        if (true) {
            if (opts.verbose) {
                cout << "Validation Epoch" << endl;
            }
            model->eval();
            float val_loss = 0;
            float sem_loss = 0;
            float r_loss = 0;
            int count = 0;
            torch::NoGradGuard no_grad;
            auto val_start = std::chrono::steady_clock::now();

            int val_count = 0;

            for (const auto& batch : *val_loader) {
                if (opts.verbose) {
                    printProgressBar(count, val_size.value(), 75);
                }
             
                iteration_val = batch.size() + iteration_val;

                std::vector<torch::Tensor> batch_data;
                std::vector<torch::Tensor> batch_radial_1;
                std::vector<torch::Tensor> batch_radial_2;
                std::vector<torch::Tensor> batch_radial_3;
                std::vector<torch::Tensor> batch_sem_target;

                for (const auto& example : batch) {
                    batch_data.push_back(example.data());
                    batch_radial_1.push_back(example.rad_1());
                    batch_radial_2.push_back(example.rad_2());
                    batch_radial_3.push_back(example.rad_3());
                    batch_sem_target.push_back(example.sem_target());

                }
		
                auto img = torch::stack(batch_data, 0);
                auto rad_1 = torch::stack(batch_radial_1, 0);
                auto rad_2 = torch::stack(batch_radial_2, 0);
                auto rad_3 = torch::stack(batch_radial_3, 0);
                auto sem_target = torch::stack(batch_sem_target, 0);
                
                img = img.to(device);
                rad_1 = rad_1.to(device);
                rad_2 = rad_2.to(device);
                rad_3 = rad_3.to(device);
                sem_target = sem_target.to(device);

                torch::Tensor output = model->forward(img);

                if ((val_count == 0) && (opts.verbose) && (session_iterations < 2) && (device == torch::kCUDA)) {
                    cout << "\r" << string(100, ' ') << "\r";
                    if (session_iterations == 0) {
                        cout << "Validation Training GPU memory profile before backpropigation and gradients loaded: " << endl;
                    }
                    else {
						cout << "Validation Training GPU memory profile with backpropigation and gradients loaded: " << endl;
					}
                    printGPUmem();
                }


                auto score_rad_1 = output.index({ torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);
                auto score_rad_2 = output.index({ torch::indexing::Slice(), 1, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);
                auto score_rad_3 = output.index({ torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);
                auto score_sem = output.index({ torch::indexing::Slice(), 3, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze(1);

                val_count++;
                //score_sem = torch::sigmoid(score_sem);

                auto loss_s = loss_sem(score_sem, sem_target);
                auto loss_r = compute_r_loss(score_rad_1, rad_1);
                loss_r += compute_r_loss(score_rad_2, rad_2);
                loss_r += compute_r_loss(score_rad_3, rad_3);

                auto loss = loss_r + loss_s;

                //cout << "Loss_r: " << loss_r.item<float>() << " Loss_s: " << loss_s.item<float>() << "\r";

                count = batch.size() + count;
                if (loss.numel() == 0)
                    std::runtime_error("Loss is empty");

                val_loss += loss.item<float>();
                sem_loss += loss_s.item<float>();
                r_loss += loss_r.item<float>();
            }


            auto val_end = std::chrono::steady_clock::now();
            auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(val_end - val_start);

            if (opts.verbose) {
                cout << "\r" << string(80, ' ') << "\r";
                cout << "Validation Time: " << val_duration.count() << " s" << endl;
            }

            val_loss /= val_size.value();
            sem_loss /= val_size.value();
            r_loss /= val_size.value();
            float mean_acc = val_loss;

            if (!opts.verbose) {
                cout << "Epoch : " << epoch << endl;
            }

            cout << "Mean Loss: " << mean_acc << endl;
            if (opts.verbose) {
                cout << "\tSemantic Loss: " << sem_loss << endl;
                cout << "\tRadial Loss: " << r_loss << endl;
            }
            bool is_best = mean_acc < best_acc_mean;

            if (is_best) {
                best_acc_mean = mean_acc;
                epochs_without_improvement = 0;
            }
            else {
                epochs_without_improvement++;
            }
            if (opts.verbose) {
                cout << "Iterations: " << iteration << endl;
                if (epochs_without_improvement > 7) {
                    cout << "Epochs without improvement: " << epochs_without_improvement << endl;
                }
            }

            //Log epoch data
            outFile << "Mean Loss: " << mean_acc << endl;
            outFile << "- Semantic Loss: " << sem_loss << endl;
            outFile << "- Radial Loss: " << r_loss << endl;
            outFile << "Iterations: " << iteration << endl;
            outFile << "Epochs without improvement: " << epochs_without_improvement << endl;
            outFile.flush();


            //================================================================\\
            //                  Save Model and Optimizer                      \\
            
            
            try {
                std::string save_location;
                if (is_best) {
                    if (opts.verbose) {
                        cout << "Saving New Best Model" << endl;
                    }
                    save_location = out + "/model_best";
                }
                else {
                    if (opts.verbose) {
                        cout << "Saving Current Model" << endl;
                    }
                    save_location = out + "/current";
                }

                if (!std::filesystem::is_directory(save_location))
                    std::filesystem::create_directory(save_location);

                torch::serialize::OutputArchive output_model_info;
                output_model_info.write("epoch", epoch);
                output_model_info.write("iteration", iteration);
                output_model_info.write("arch", model->name());
                output_model_info.write("best_acc_mean", best_acc_mean);
                output_model_info.write("loss", val_loss);
                output_model_info.write("optimizer", opts.optim);
                output_model_info.write("lr", current_lr);

                output_model_info.save_to(save_location + "/info.pt");

                torch::serialize::OutputArchive output_model_archive;
                model->to(torch::kCPU);
                model->save(output_model_archive);
                model->to(device);
                output_model_archive.save_to(save_location + "/model.pt");


                torch::serialize::OutputArchive output_optim_archive;
                optim->save(output_optim_archive);
                output_optim_archive.save_to(save_location + "/optim.pt");
            }
            catch (std::exception& e) {
                std::cout << "Error saving model: " << e.what() << std::endl;
            }
            
        }
        
        // Reduce learning rate every 70 epoch
        if (opts.reduce_on_plateau == false){
            if (epoch % 70 == 0 && epoch != 0) {
                cout << "Learning rate reduction" << endl;
                current_lr.clear();
                for (auto& param_group : optim->param_groups()) {
                    if (param_group.has_options()) {
                        //double lr = param_group.options().get_lr();
                        double lr = static_cast<torch::optim::AdamOptions &>(param_group.options()).lr();
			cout << "Current LR: " << lr << endl;
                        double new_lr = lr * 0.1;
                        if (opts.optim == "adam") 
                            static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(new_lr);
                        if (opts.optim == "sgd")
                            static_cast<torch::optim::SGDOptions &>(param_group.options()).lr(new_lr);
                        cout << "New LR: " << new_lr << endl;
                        current_lr.push_back(new_lr);
                    }
                    else {
                        cout << "Error: param_group has no options" << endl;
                    }
                }
            }
        }
        if (opts.reduce_on_plateau && epochs_without_improvement >= opts.patience) {
            cout << "Reducing learning rate" << endl;
            outFile << "Reducing learning rate" << endl;
            outFile.flush();

            current_lr.clear();
            for (auto& param_group : optim->param_groups()) {
                if (param_group.has_options()) {
		    double lr = static_cast<torch::optim::AdamOptions &>(param_group.options()).lr();
                    //double lr = param_group.options().get_lr();
                    cout << "Current LR: " << lr << endl;
                    double new_lr = lr * 0.1;
                    if (opts.optim == "adam")
                        static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(new_lr);
                    else if (opts.optim == "sgd")
                        static_cast<torch::optim::SGDOptions&>(param_group.options()).lr(new_lr);
                    else
                        cout << "Error: Invalid optimizer" << endl;
                    cout << "New LR: " << new_lr << endl;
                    current_lr.push_back(new_lr);
                } else {
                    cout << "Error: param_group has no options" << endl;
                } 
            }
            epochs_without_improvement = 0;
        }
        if (iteration >= max_iteration) {
            break;
        }

        auto epoch_train_end = std::chrono::steady_clock::now();
        auto epoch_total_time = std::chrono::duration_cast<std::chrono::seconds>(epoch_train_end - epoch_start_time);

        auto total_train_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_train_end - total_train_start);
        float average_epoch_time = total_train_duration.count() / (epoch + 1 - starting_epoch);
  
        if (opts.verbose) {
            cout << "Epoch Training Time: " << epoch_total_time.count() << " s" << endl;
            cout << "Average Time per Epoch: " << average_epoch_time << " s" << endl;
        }
        else {
            if (epoch % 10 == 0) {
                cout << "Average Time per Epoch: " << average_epoch_time << " s" << endl;
            }
        }
        cout << endl;

        epoch++;
        session_iterations++;
    }
    outFile.close();
}

torch::Tensor Trainer::compute_r_loss(torch::Tensor pred, torch::Tensor gt) {
    torch::Tensor gt_mask = gt != 0;
    torch::Tensor gt_masked = torch::masked_select(gt, gt_mask);
    
    torch::Tensor pred_masked = torch::masked_select(pred, gt_mask);
    
    // Compute the loss
    torch::Tensor loss = loss_radial(pred_masked, gt_masked);
    
    //cout<<loss.item<float>()<<"  ,  "<<static_cast<float>(gt_masked.size(0))<<endl;
    // Normalize the loss
    if (static_cast<float>(gt_masked.size(0)) != 0)
    	loss = loss / static_cast<float>(gt_masked.size(0));

    return loss;
}


void Trainer::printGPUmem() {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    size_t used_memory = total_memory - free_memory;
    cout << "\tUsed GPU memory: " << used_memory / (1024 * 1024 * 1024) << " GB, "
        << (used_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
        << used_memory % (1024 * 1024) << " bytes" << endl;

    cout << "\tFree GPU memory: " << free_memory / (1024 * 1024 * 1024) << " GB, "
        << (free_memory % (1024 * 1024 * 1024)) / (1024 * 1024) << " MB, "
        << free_memory % (1024 * 1024) << " bytes" << endl;
}


cv::Mat Trainer::tensor_to_mat(torch::Tensor tensor) {
    int rows = tensor.size(0);
    int cols = tensor.size(1);
    cv::Mat mat(rows, cols, CV_32FC1, tensor.data_ptr<float>());
    return mat.clone();
}
