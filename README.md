<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>
<div align="center">

# 🚀 AI Lunar Landing
### A Reinforcement Learning Approach to Master Lunar Landing
<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>

![Algorithm](https://img.shields.io/badge/Algorithm-Reinforcement%20Learning-blue)
![Framework](https://img.shields.io/badge/PyTorch-1.12-orange)
![Colab](https://img.shields.io/badge/Colab-Optimized-yellow)

![Lunar Landing AI](https://img.shields.io/badge/AI-LunarLanding-green)

![download1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/26c77322-4db0-471c-a11e-f7fee9e4af08)


</div>

<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>

## 🌌 Project Overview

This project implements an **AI agent** capable of mastering the **Lunar Landing** task using **Reinforcement Learning** techniques. The agent learns to land on the lunar surface safely by maximizing rewards through trial and error in a simulated environment.

<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>

## 🛠️ Key Features

- **Reinforcement Learning Framework**: Utilizes advanced algorithms to train the agent for optimal landing strategies.
- **Optimized for Google Colab**: Easy to use and experiment with a fully integrated Colab setup.
- **Custom Environment**: Built-in support for the Lunar Landing environment to enhance training effectiveness.
- **Dynamic Visualizations**: Real-time visual feedback of the agent's performance during training.
- **Reward-Based Learning**: Implements reward mechanisms to guide the agent towards successful landings.

<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>

## 🎯 Tech Stack

<div style="display: flex; align-items: center;">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="60" height="60">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="60" height="60">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/googlecloud/googlecloud-original.svg" alt="Google Cloud" width="60" height="60">
    <img src="https://gymnasium.farama.org/_images/gymnasium-text.png" alt="Gymnasium" width="60" height="60">
</div>

- **PyTorch 1.12**
- **Gymnasium 0.29.1**
- **CUDA Acceleration**
- **Multi-threading Support**

<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>

## 💡 Implementation Highlights

1. **Neural Network Architecture**: A convolutional neural network (CNN) processes the input states, allowing the agent to learn effective landing strategies.
   ```python
   class LunarLandingAgent(nn.Module):
       def __init__(self, action_size):
           super(LunarLandingAgent, self).__init__()
           self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
           self.flatten = nn.Flatten()
           self.fc1 = nn.Linear(64 * 9 * 9, 128)
           self.fc2 = nn.Linear(128, action_size)

    Training Strategy: Utilizes an optimized training loop with reward shaping to improve the learning process.

    python

    def train_agent(agent, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state

    Dynamic Visualization: Real-time plotting of the agent's landing performance, including success rates and reward accumulation.

<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>
📈 Performance

    Total Episodes: 1000
    Maximum Landing Success Rate: 95%
    Training Duration: Approximately 5 hours (on GPU)
    Reward Optimization: Implemented advantage-based updates for efficient learning.

<img width="2000rem" src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp"><br>
✨ Future Enhancements

    Expand to Other Environments: Implement similar algorithms for different landing tasks or environments.
    Real-time Analytics: Integrate TensorBoard for enhanced performance tracking.
    3D Visualization: Create a 3D simulation environment for improved training feedback.

<div align="center">

Built with 💪 using PyTorch & Gymnasium
</div> ```
Key Points:

    This template includes sections for project overview, features, tech stack, implementation highlights, performance metrics, and future enhancements.
    You can modify the code snippets, performance data, or future plans as necessary to fit your project specifics.
    Ensure that you have images or assets ready if you want to replace the placeholders.
