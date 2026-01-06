🧠 LifeOS: Personal Knowledge Distillation via Llama-3

"Transforming a static schedule into a reasoning engine."

📖 Overview

LifeOS is a specialized Large Language Model (LLM) fine-tuned to act as a hyper-personalized health and productivity assistant.

Unlike generic chatbots, LifeOS is trained on a specific Knowledge Graph containing my academic calendar, workout protocols, and health logic. It utilizes Synthetic Data Generation to expand a small seed dataset into 10,000+ high-quality training samples, preventing overfitting while ensuring logic retention.

🛠 Tech Stack

Model: Llama-3 (8B) via Unsloth (for 2x faster training).

Technique: QLoRA (Quantized Low-Rank Adaptation).

Data Engineering: Python-based synthetic data factory with combinatorial logic.

Hardware: Trained on NVIDIA T4 GPU (Google Colab).

📊 Data Engineering (The "Math" Part)

To solve the "Data Scarcity" problem (having only ~50 real-life data points), I engineered a probabilistic data generator.

It uses combinatorial templates to mix-and-match:

Intents (Schedule, Health, Diet)

Tones (Clinical, Motivational, Casual)

Contexts (Exam week, Rest day, Injury)

🚀 Quick Start

1. Installation

Clone the repo and install dependencies:

git clone
pip install -r requirements.txt


2. Data Generation

Generate synthetic training data from the seed file:

python src/data_generator.py


This will create data/generated_data_10k.json.

3. Visualization

Generate data distribution plots to verify balance:

python src/visualizer.py


4. Training

Run the training script (optimized for Colab/Jupyter):

# Recommended to run in Google Colab for GPU support
python notebook/life_os_training.py


5. Web Demo

Launch a local chatbot interface:

python app_demo.py


📈 Performance

Training Loss: Converged from 2.8 to 0.9 in 60 steps.

Qualitative Result: The model successfully handles complex logic like "I have shoulder pain" by recommending specific rehab exercises (Dead Hangs) sourced from the medical knowledge base.
