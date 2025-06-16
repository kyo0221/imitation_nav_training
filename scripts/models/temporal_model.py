import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConditionalAnglePredictor(nn.Module):
    def __init__(self, n_channel, n_out, input_height, input_width, n_action_classes, 
                 sequence_length=10, hidden_size=256, num_layers=2, prediction_horizon=3,
                 rnn_type='LSTM'):
        """
        時系列情報を考慮した条件付き角度予測モデル
        
        Args:
            n_channel: 入力チャンネル数
            n_out: 出力次元数
            input_height: 入力画像の高さ
            input_width: 入力画像の幅
            n_action_classes: アクションクラス数
            sequence_length: 時系列の長さ
            hidden_size: RNNの隠れ状態次元数
            num_layers: RNNの層数
            prediction_horizon: 未来予測のホライズン
            rnn_type: RNNのタイプ ('LSTM' or 'GRU')
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.n_action_classes = n_action_classes
        self.rnn_type = rnn_type

        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout_conv = nn.Dropout2d(p=0.2)
        self.dropout_fc = nn.Dropout(p=0.5)

        # CNNエンコーダ部分（単一フレーム用）
        def conv_block(in_channels, out_channels, kernel_size, stride, apply_bn=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels) if apply_bn else nn.Identity(),
                self.relu,
                self.dropout_conv
            ]
            return nn.Sequential(*layers)

        self.conv1 = conv_block(n_channel, 32, kernel_size=5, stride=2)
        self.conv2 = conv_block(32, 48, kernel_size=3, stride=1)
        self.conv3 = conv_block(48, 64, kernel_size=3, stride=2)
        self.conv4 = conv_block(64, 96, kernel_size=3, stride=1)
        self.conv5 = conv_block(96, 128, kernel_size=3, stride=2)
        self.conv6 = conv_block(128, 160, kernel_size=3, stride=1)
        self.conv7 = conv_block(160, 192, kernel_size=3, stride=1)
        self.conv8 = conv_block(192, 256, kernel_size=3, stride=1)

        # CNN特徴量の次元を計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channel, input_height, input_width)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.flatten(x)
            self.cnn_feature_size = x.shape[1]

        self.cnn_layer = nn.Sequential(
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7, self.conv8,
            self.flatten
        )

        # 特徴量結合層（CNN特徴量 + アクション情報）
        self.feature_fusion = nn.Linear(self.cnn_feature_size + n_action_classes, hidden_size)

        # 時系列処理層（LSTM/GRU）
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.3 if num_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.3 if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # 現在制御量予測ヘッド
        self.current_predictor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            self.relu,
            self.dropout_fc,
            nn.Linear(256, n_out)
        )

        # 未来制御量予測ヘッド（複数時点）
        self.future_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 256),
                self.relu,
                self.dropout_fc,
                nn.Linear(256, n_out)
            ) for _ in range(prediction_horizon)
        ])

        # 条件付き学習用のブランチ
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 256),
                self.relu,
                nn.Linear(256, n_out)
            ) for _ in range(n_action_classes)
        ])

    def forward(self, sequence_images, sequence_actions, return_future=False):
        """
        前向き推論
        
        Args:
            sequence_images: [batch_size, seq_len, C, H, W]
            sequence_actions: [batch_size, seq_len, action_dim]
            return_future: 未来予測も返すかどうか
        
        Returns:
            current_pred: 現在の制御量予測
            future_preds: 未来の制御量予測（return_future=Trueの場合）
        """
        batch_size, seq_len = sequence_images.shape[:2]
        
        # 各フレームをCNNで特徴抽出
        sequence_features = []
        for t in range(seq_len):
            # 単一フレームの特徴抽出
            frame_features = self.cnn_layer(sequence_images[:, t])  # [batch_size, cnn_feature_size]
            
            # アクション情報と結合
            combined_features = torch.cat([frame_features, sequence_actions[:, t]], dim=1)
            fused_features = self.relu(self.feature_fusion(combined_features))
            
            sequence_features.append(fused_features)
        
        # 時系列特徴量を結合 [batch_size, seq_len, hidden_size]
        sequence_features = torch.stack(sequence_features, dim=1)
        
        # RNNで時系列処理
        rnn_output, _ = self.rnn(sequence_features)
        
        # 最後の時点の隠れ状態を使用
        last_hidden = rnn_output[:, -1]  # [batch_size, hidden_size]
        
        # 現在の制御量予測（条件付き）
        current_action = sequence_actions[:, -1]  # 最後のアクション
        action_indices = torch.argmax(current_action, dim=1)
        
        current_pred = torch.zeros(batch_size, self.branches[0][-1].out_features, 
                                  device=sequence_images.device, dtype=last_hidden.dtype)
        
        for idx, branch in enumerate(self.branches):
            selected_idx = (action_indices == idx).nonzero().squeeze(1)
            if selected_idx.numel() > 0:
                current_pred[selected_idx] = branch(last_hidden[selected_idx])
        
        if not return_future:
            return current_pred
        
        # 未来予測
        future_preds = []
        for i, predictor in enumerate(self.future_predictors):
            future_pred = predictor(last_hidden)
            future_preds.append(future_pred)
        
        return current_pred, torch.stack(future_preds, dim=1)  # [batch_size, prediction_horizon, n_out]


class TemporalLoss(nn.Module):
    def __init__(self, current_weight=1.0, future_weight=0.5, temporal_smoothness_weight=0.1):
        """
        時系列予測用の複合損失関数
        
        Args:
            current_weight: 現在予測の重み
            future_weight: 未来予測の重み  
            temporal_smoothness_weight: 時間的平滑性の重み
        """
        super().__init__()
        self.current_weight = current_weight
        self.future_weight = future_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, current_pred, future_preds, current_target, future_targets):
        # 現在予測の損失
        current_loss = self.mse_loss(current_pred, current_target)
        
        # 未来予測の損失
        future_loss = 0
        if future_preds is not None and future_targets is not None:
            for i in range(future_preds.shape[1]):
                future_loss += self.mse_loss(future_preds[:, i], future_targets[:, i])
            future_loss /= future_preds.shape[1]
        
        # 時間的平滑性の損失（未来予測間の差分を小さくする）
        smoothness_loss = 0
        if future_preds is not None and future_preds.shape[1] > 1:
            for i in range(future_preds.shape[1] - 1):
                smoothness_loss += self.mse_loss(future_preds[:, i+1], future_preds[:, i])
            smoothness_loss /= (future_preds.shape[1] - 1)
        
        total_loss = (self.current_weight * current_loss + 
                     self.future_weight * future_loss + 
                     self.temporal_smoothness_weight * smoothness_loss)
        
        return total_loss, current_loss, future_loss, smoothness_loss