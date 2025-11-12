
from xgboost import XGBClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F



def xgboost_model(num_classes):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False
    )
    return model


def svm_model():
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    return model


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=14, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, 14, 55]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def cnn_model(num_classes):
    return CNN(num_classes=num_classes)


class LSTM(nn.Module):
    def __init__(self, num_classes):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=14,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def lstm_model(num_classes):
    return LSTM(num_classes=num_classes)


if __name__ == "__main__":
    num_classes = 26
    input_shape = (55, 14)

    xgb = xgboost_model(num_classes=num_classes)
    svm = svm_model()
    cnn = cnn_model(num_classes=num_classes)
    lstm = lstm_model( num_classes=num_classes)


