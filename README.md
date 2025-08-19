# PopSim: A Novel Simulation-based Paradigm for Social Media Popularity Prediction

## 🤖 Introduction
PopSim introduces a simulation-based paradigm for Social Media Popularity Prediction (SMPP). Unlike traditional feature-engineering approaches, PopSim leverages the reasoning and generative capabilities of large language models (LLMs) to simulate the propagation dynamics of user-generated content (UGC), improving prediction accuracy. It incorporates agent-based modeling to simulate interactions among users and UGC in social networks, capturing the complex, nonlinear dynamics of content propagation.

## 💡 Features
* **Simulation-based Popularity Prediction:** Uses LLM-based multi-agent interactions to simulate UGC propagation dynamics.
* **Social Mean Field Simulation:** Models agent interactions within the network to improve prediction accuracy.
* **Multimodal Contextualization:** Transforms heterogeneous metadata (images, text, etc.) into enriched, unified text for LLM processing.
* **State-of-the-art Performance:** Outperforms feature-engineering and retrieval-based methods in SMPP tasks.

## 🚀 Getting Started
### Requirements
Make sure you have Python >= 3.9 and the following dependencies:

* `Python >= 3.9`
* `torch == 2.5.1`
* `Mesa == 2.2.4`
* `transformers == 4.52.4`
* `sentence_transformers == 4.1.0`


### Installation
1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Follow the instructions in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to set up the environment. This is required as our training and inference processes are based on this library.

### Data Preparation

1. **Download the Dataset**  
   Download the dataset from the following link: [SMP Challenge Dataset](https://smp-challenge.com/dataset)

2. **Place the Dataset**  
   After downloading the dataset, place it into the directory: `/data/meta_data/`

3. **Preprocess the Data**  
   Run the `data_preprocess.py` script to convert the dataset into the JSON format. This will prepare the dataset for further processing:
   ```bash
   python data_preprocess.py
   ```
4. **Split the Data**  
   After preprocessing, run the data_split.py script to serialize the data and perform dataset splitting (training, validation, and test sets):
   ```bash
   python data_split.py
   ```

After completing these steps, your dataset will be properly prepared for use in simulations and model training.


### Running the Simulation

1. **Set up the Configuration File**  
   Before running the simulation, make sure to configure the `simulation/config.yaml` file. In this file, you'll need to set the following parameters:
   - **Model Parameters**: Specify the paths to your pre-trained models, including the LLM model and the embedding model, as well as other model-related parameters.
   - **Dataset Parameters**: Set the paths to your prepared dataset, including raw simulation data and user profile data.
   - **Simulation Parameters**: Configure key simulation settings, such as the number of agents, the number of simulation rounds, and other relevant settings.


2. **Run the Simulation**  
   Once the configuration file is set up, run the simulation by executing the following command:
   ```bash
   python SMF_simulator.py
   ```

The simulation results will be saved in the `simulation_result` folder.


### Training the prediction agent
1. **Build the Training Data**  
   First, move the simulation results to the appropriate directory:
   ```bash
   mv simulation_result /data/meta_data/
   ```

   Run the MF_data_construct.py script to construct the training data for the prediction agent:
   ```bash
   python MF_data_construct.py
   ```
   This will generate the training dataset, which should then be moved to /data/:
   ```bash
   mv generated_dataset /data/
   ```

2. **Modify Training Parameters**  
   Edit the training parameters in `examples/train_lora/llama3_lora_sft_demo.yaml` to configure the training process.

3. **Train the Model**  
   Once the training parameters are set, run the following command to train the model using LLaMAFactory:
   ```bash
   llamafactory-cli train examples/train_lora/llama3_lora_sft_demo.yaml
   ```

4. **Merge the Trained Model**  
   After training, merge the LoRA-trained model by running:
   ```bash
   llamafactory-cli export examples/merge_lora/llama3_lora_sft_demo.yaml
   ```

5. **Run Inference**  
   Finally, run `inference.py` to perform inference with the trained model:
   ```bash
   python inference.py
   ```

Our training and model merging process follows the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) guidelines. Please refer to their documentation for more details.



# 🤝 Contributing
We welcome contributions from the community! Please fork the repository and submit a pull request for any improvements or bug fixes. For large changes, please discuss the proposal first by opening an issue.

# 📝 License
PopSim is distributed under the [MIT License](./LICENSE). Feel free to use and modify it for your own projects.

# 📄 Cite
More details will be released soon.
