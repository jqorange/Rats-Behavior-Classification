# Function Reference

Below is a high level summary of the public functions and class methods found in the repository. They are grouped by file for easy lookup.

## evaluate.py
* **load_valid_segments** – read `results.txt` and return a mapping of valid label segments for each session.
* **evaluate_predictions** – compute accuracy, precision, recall and F1 for each label column.
* **load_labels** – read ground truth CSV labels for one session.
* **load_predictions** – slice prediction CSVs using segment indices.
* **main** – command line entry for batch evaluation of prediction files.

## inference.py
* **latest_checkpoint** – locate the most recent encoder checkpoint.
* **main** – generate representations for one or more sessions.

## models/classifier.py
* **MLPClassifier.__init__** – build a small two layer MLP.
* **MLPClassifier.forward** – apply the classifier to an input tensor.

## models/deep_mlp.py
* **DeepMLPClassifier.__init__** – construct a deeper MLP with configurable hidden sizes.
* **DeepMLPClassifier.forward** – run the classifier.

## models/dilated_conv.py
* **SamePadConv.__init__** – convolution with automatic same padding.
* **SamePadConv.forward** – run the convolution and trim extra padding.
* **ConvBlock.__init__** – residual block of two convolutions.
* **ConvBlock.forward** – forward pass with GELU activations.
* **DilatedConvEncoder.__init__** – stack multiple residual blocks with increasing dilation.
* **DilatedConvEncoder.forward** – encode a sequence.

## models/domain_adapter.py
* **_build_align_block** – helper that creates a small alignment MLP.
* **DomainAdapter.__init__** – projection layer with optional session embedding.
* **DomainAdapter.set_mode** – choose between `aware`, `align` or `none` modes.
* **DomainAdapter.forward** – apply the adapter to input features.

## models/encoder.py
* **Encoder.__init__** – build the domain adapter, dilated conv stack and Transformer layer.
* **Encoder.forward** – encode a sequence with optional masking.

## models/fusion.py
* **EncoderFusion.__init__** – create two encoders and a cross‑attention fusion module.
* **EncoderFusion.forward** – produce the fused representation for a pair of sequences.

## models/losses.py
* **hierarchical_contrastive_loss** – instance and temporal contrastive loss across multiple scales.
* **instance_contrastive_loss** – contrast between samples within the same batch.
* **temporal_contrastive_loss** – contrast across time within each sequence.
* **positive_only_supcon_loss** – supervised contrastive loss that only attracts positives.
* **multilabel_supcon_loss_bt** – multi‑label supervised contrastive loss with pooling.
* **compute_contrastive_losses** – dispatch supervised or unsupervised contrastive losses depending on stage.
* **CenterLoss.__init__** – maintain class centroids.
* **CenterLoss.forward** – compute distance from features to class centers.
* **UncertaintyWeighting.__init__** – learn weights for combining multiple losses.
* **UncertaintyWeighting.forward** – apply uncertainty weighting.
* **PrototypeMemory.__init__** – store class prototypes used for pseudo labelling.
* **PrototypeMemory.assign_labels** – assign pseudo labels based on cosine similarity.
* **PrototypeMemory.update** – update prototypes using labelled or pseudo labelled features.
* **PrototypeMemory.forward** – soft alignment loss between features and prototypes.
* **prototype_repulsion_loss** – penalise similarity to incorrect prototypes. Supports soft labels.
* **prototype_center_loss** – center loss computed using class prototypes. Supports soft labels.

## models/masking.py
* **generate_continuous_mask** – drop contiguous blocks of frames.
* **generate_binomial_mask** – sample frames independently using a Bernoulli distribution.

## multi_class_classifier.py
* **load_valid_segments** – parse `results.txt` for valid label regions.
* **load_session_data** – load features and labels for a session.
* **load_unlabeled_data** – return data outside labelled segments for self training.
* **build_datasets** – assemble training, test and unlabeled datasets.
* **evaluate** – compute accuracy and F1 for a data loader.
* **pseudo_label** – generate pseudo labels from high confidence predictions.
* **self_training** – iterative pseudo label training loop.
* **main** – command line interface for self training.

## multi_class_inference.py
* **load_representation** – load a representation file from disk.
* **run_inference** – run a saved MLP on one session and save predictions.
* **main** – script entry point.

## plot_embeddings.py
* **load_channel_names** – read channel names for plotting.
* **compute_cluster_metrics** – evaluate cluster purity in PCA space.
* **main** – draw PCA plot from representation files.

## predict_repr_mlp.py
* **load_model** – load a representation‑level MLP.
* **main** – perform inference on multiple sessions.

## train.py
* **main** – create a `TrainPipline` and run the training workflow.

## train_repr_mlp.py
* **load_valid_segments** – parse `results.txt` for valid label regions.
* **load_session_data** – load representations and labels for supervised training.
* **load_unlabeled_data** – load unlabeled representations.
* **build_datasets** – prepare training and test sets with optional unlabeled data.
* **evaluate** – compute accuracy and F1 of the deep MLP.
* **pseudo_label** – select high confidence samples as pseudo labels.
* **self_training** – multi‑step self training loop.
* **main** – command line entry.

## utils/TrainPipline.py
* **TrainPipline.__init__** – set paths, sessions and basic parameters.
* **TrainPipline._setup_device** – choose CPU or CUDA device.
* **TrainPipline.load_and_prepare_data** – load, truncate and align all sessions.
* **prepare_session** – inner helper used during loading.
* **TrainPipline.initialize_trainer** – create the `FusionTrainer` with given hyper parameters.
* **TrainPipline.test_model_components** – sanity check that the model runs on a few samples.
* **TrainPipline.train_model** – run the three‑stage training using the trainer.
* **TrainPipline.evaluate_model** – evaluate the trained model on the test set.
* **TrainPipline.run_full_pipeline** – convenience function executing the entire workflow.

## utils/data_loader.py
* **DataLoader.__init__** – configure paths for IMU, DLC and label files.
* **DataLoader.load_original_data** – read unsupervised IMU and DLC arrays.
* **DataLoader.load_supervised_data** – read supervised segments and labels.
* **DataLoader.load_all_data** – call both loading functions and return a summary.
* **DataLoader.get_data_summary** – return dictionaries of all loaded arrays.
* **DataLoader.get_session_data** – fetch data for a single session.
* **DataLoader.print_data_info** – print statistics about every session.

## utils/tools.py
* **take_per_row** – slice a batch of sequences starting at different offsets.
* **take_per_row_safe** – safe version that pads beyond sequence ends.
* **take_per_row_vectorized** – vectorised slicing implementation.
* **test_take_per_row** – demonstration of the slicing functions.

## utils/trainer.py
* **FusionTrainer.__init__** – build encoders, classifier and optimisers.
* **train_contrastive_phase** – run a contrastive learning stage.
* **train_stage3** – stage 3 training with prototype loss.
* **train_contrastive_multi_session** – unsupervised training mixing sessions per batch.
* **train_mlp_phase** – supervised MLP training after the encoders are frozen.
* **fit** – execute the full three‑stage curriculum.
* **init_stage2** – load adapters and freeze encoders for stage 2.
* **init_stage3** – prepare prototype memory for stage 3.
* **encode** – encode IMU and DLC sequences without classification.
* **encode_state1** – encode using the session‑aware adapters from stage 1.
* **predict** – run the MLP classifier on new data.
* **save** – save encoder and classifier checkpoints.
* **load** – load checkpoints.
* **load_stage2** – load only the encoder for stage 2 inference.