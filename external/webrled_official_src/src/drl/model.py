import torch
import torch.nn as nn
import torch.nn.functional as F


# Encording
class Encoder(nn.Module):
    """
    Encoder with convolution
    Attributes:
      conv1   : first convolutional layer
      conv2   : second convolutional layer 
      conv3   : third convolutional layer
      flatten : for tensor to be flatten
      fc      : full connected layer
    """

    def __init__(self, units, input_dim=768, hidden=512):
        super(Encoder, self).__init__()
        """
        units     (int): dim of output
        input_dim (int): dim of input
        hiddne    (int): dim of hidden
        """

        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, units)

    def forward(self, x):
        """
        Args:
          x (torch.tensor): input [b, 768]
        Returns
          x (torch.tensor): ouput [b, units]
        """

        x = F.relu(self.fc1(x))  # (b, 512)
        x = self.fc2(x)  # (b, units)
        return x


# intrinsic or extrinsic QNetwork
class QNetwork(nn.Module):
    """
    Attributes:
      action_space (int): dim of action space
      num_arms     (int): number of arms used in multi-armed bandit problem
      encoder           : encoder
      lstm              : LSTM layer
      fc                : fully connected layer
      fc_adv            : fully connected layer to get advantage
      fc_v              : fully connected layer to get value
    """

    def __init__(self, action_space, input_dim=768, hidden=512, units=512, num_arms=32):
        super(QNetwork, self).__init__()
        """
        Args:
          action_space (int): dim of action space
          input_dim (int): dim of input
        """

        self.action_space = action_space
        self.num_arms = num_arms

        self.encoder = Encoder(units, input_dim)
        self.lstm = nn.LSTM(input_size=units + self.action_space + self.num_arms + 2,
                            hidden_size=hidden,
                            batch_first=False)

        self.fc = nn.Linear(hidden, hidden)
        self.fc_adv = nn.Linear(hidden, action_space)
        self.fc_v = nn.Linear(hidden, 1)

    def forward(self, input, states, prev_action, prev_in_rewards, prev_ex_rewards, j):
        """
        Args:
          input           (torch.tensor): state [b, 768]
          prev_action     (torch.tensor): previous action [b, action_space]
          prev_in_rewards (torch.tensor): previous intrinsic reward [b]
          prev_ex_rewards (torch.tensor): previous extrinsic reward [b]
        """

        # (b, q_units)
        x = F.relu(self.encoder(input))

        # (b, action_space)
        # prev_action_onehot = F.one_hot(prev_action, num_classes=self.action_space)
        prev_action_onehot = prev_action

        # (b, num_arms)
        j_onehot = F.one_hot(j, num_classes=self.num_arms)

        # (b, q_units+action_space+num_arms+2)
        x = torch.cat([x, prev_action_onehot, prev_in_rewards[:, None], prev_ex_rewards[:, None], j_onehot], dim=1)

        # (1, b, hidden)
        x, states = self.lstm(x.unsqueeze(0), states)

        # (b, action_space)
        A = self.fc_adv(x.squeeze(0))

        # (b, 1)
        V = self.fc_v(x.squeeze(0))

        # (b, action_space)
        Q = V.expand(-1, self.action_space) + A - A.mean(1, keepdim=True).expand(-1, self.action_space)

        return Q, states


class EmbeddingNet(nn.Module):
    """
    Attributes
      encoder : encoder
    """

    def __init__(self, input_dim=768, units=128):
        super(EmbeddingNet, self).__init__()
        """
        Args:
          input_dim (int): dim of input
        """

        self.encoder = Encoder(units, input_dim)

    def forward(self, inputs):
        """
        Args:
          input (torch.tensor): state [b, 768]
        Returns:
          embeded state [b, emebed_units]
        """

        return F.relu(self.encoder(inputs))


class EmbeddingClassifer(nn.Module):
    """
    Attributes:
      fc1 : fully connected layer
      fc2 : fully connected layer to get action probability
    """

    def __init__(self, action_space, hidden=128):
        super(EmbeddingClassifer, self).__init__()
        """
        Args:
          action_space (int): dim of action space
        """

        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, action_space)

    def forward(self, input1, input2):
        """
        Args:
          embeded state (torch.tensor): state [b, emebed_units]
        Returns:
          action probability [b, action_space]
        """

        x = torch.cat([input1, input2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x


class LifeLongNet(nn.Module):
    """
    Attributes
      conv_encoder : convolutional encoder
    """

    def __init__(self, input_dim=768, units=128):
        super(LifeLongNet, self).__init__()
        """
        Args:
          input_dim (int): dim of input
        """

        self.encoder = Encoder(units, input_dim)

    def forward(self, inputs):
        """
        Args:
          input (torch.tensor): state [b, 768]
        Returns:
          lifelong state [b, lifelong_units]
        """

        return self.encoder(inputs)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 768 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 768
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LongAutoEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 768 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 9)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 768
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_dim),
            # torch.nn.Sigmoid() todo try tanh
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DomElemDiscriminator(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(DomElemDiscriminator, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        return out
