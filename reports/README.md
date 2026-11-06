# sm702 冻结结果

本目录保存 sm702 在 2026-11-05 的样本外回测结果与模型，便于不重跑训练直接查看。

- `backtest_summary.csv`：止损叠加层 vs 买入持有总体指标
- `backtest_yearly.csv`：逐年指标
- `backtest_thresholds.csv`：阈值 0.15–0.40 敏感性
- `equity_curve.csv`：逐日净值
- `trades_stop.csv.gz`：逐笔止损交易
- `fold_metrics.csv`：每折训练/验证/测试区间、阈值、AUC/AP
- `yearly_signal_metrics.csv`：逐年 AUC/AP/Brier
- `feature_importance.csv`：特征重要性
- `latest_signals_2026-09-30.csv`：最新交易日止损清单
- `feature_names.json`：特征列顺序
- `models/lgbm_YYYY.txt`：2020–2026 各测试年 LightGBM 模型

数据截止 2026-09-30；默认阈值 0.25。原始行情数据库、特征矩阵、完整预测文件
体积较大，未纳入 Git 仓库；需要时按主 README 的命令重新生成。
