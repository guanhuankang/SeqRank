## Model Zoo

| Model Items           | TrainSet | Results                                                      | Checkpoints                                                  | Config                                   | SA-SOR | SOR   | MAE  |
| --------------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- | ------ | ----- | ---- |
| seqrank_assr          | ASSR     | [results.pth](assr_swinL/seqrank_assr_swinL_results.pth)     | [ckp.pth](https://drive.google.com/file/d/1rWHEUlnCWweoYqdY9JvrHuts-2lmUr_B/view?usp=drive_link) | [config](assr_swinL/config.yaml)         | 0.685  | 0.870 | 7.22 |
| seqrank_irsr          | IRSR     | [results.pth](irsr_swinL/seqrank_irsr_swinL_results.pth)     | [ckp.pth](https://drive.google.com/file/d/1PUSJLRxA9sIJoYXx0Si3TawoKvnLIU3o/view?usp=drive_link) | [config](irsr_swinL/config.yaml)         | 0.576  | 0.822 | 6.20 |
| seqrank_assr_correct* | ASSR     | [results.pth](assr_swinL_correct/seqrank_assr_swinL_correct_results.pth) | [ckp.pth](https://drive.google.com/file/d/1oaFmicXHh3kaipY4vJ76Xb23utHn2-uo/view?usp=drive_link) | [config](assr_swinL_correct/config.yaml) | 0.680  | 0.873 | 7.45 |
| seqrank_irsr_correct* | IRSR     | [results.pth](irsr_swinL_correct/seqrank_irsr_swinL_correct_results.pth) | [ckp.pth](https://drive.google.com/file/d/1e9JUHB3gVq4AK1Z4lIQvPYz6h2rs3zre/view?usp=drive_link) | [config](irsr_swinL_correct/config.yaml) | 0.584  | 0.828 | 6.09 |

*: We correct some minor mistakes in this version. Specifically, previous versions use a special pixel decoder, FrcPN, and adopt six SRMs. We correct the mistakes and switch to FPN decoder and use two SRMs. Then we retrain the model following the same training recipe on two SOR benchmarks.
