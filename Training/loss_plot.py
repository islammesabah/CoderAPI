import matplotlib.pyplot as plt
import pandas as pd

M1_loss_path = "./saved_models/exp/saved_M1_instruct_codet5p_16b_100epochs_fp16_general_prompt/loss.csv"
M2_loss_path = "./saved_models/exp/saved_M2_instruct_codet5p_16b_100epochs_fp16_general_prompt_no_decoder_inp/loss.csv"
M3_loss_path = "./saved_models/exp/saved_M3_instruct_codet5p_16b_100epochs_fp16_code_prompt/loss.csv"
M4_loss_path = "./saved_models/exp/saved_M4_instruct_codet5p_16b_100epochs_fp16_code_prompt_no_decoder_inp/loss.csv"
M5_loss_path = "./saved_models/exp/saved_M5_instruct_codet5p_16b_100epochs_fp16_docu_code_prompt/loss.csv"
M6_loss_path = "./saved_models/exp/saved_M6_instruct_codet5p_16b_100epochs_fp16_docu_code_prompt_no_decoder_inp/loss.csv"
M7_loss_path = "./saved_models/exp/saved_M7_instruct_codet5p_16b_100epochs_fp16_general_prompt_docu_encoder_inp/loss.csv"
M8_loss_path = "./saved_models/exp/saved_M8_instruct_codet5p_16b_100epochs_fp16_code_prompt_docu_encoder_inp/loss.csv"
M9_loss_path = "./saved_models/exp/saved_M9_instruct_codet5p_16b_100epochs_fp16_docu_code_prompt_docu_encoder_inp/loss.csv"

# plot loss for model with decoder input
trainer_history_general = pd.read_csv(M1_loss_path)
trainer_history_code = pd.read_csv(M3_loss_path)
trainer_history_docu = pd.read_csv(M5_loss_path)
plt.plot(trainer_history_general.epoch, trainer_history_general.loss, label='General prompt', color='blue')
plt.plot(trainer_history_general.epoch, trainer_history_code.loss, label='Coding-specific prompt', color='red')
plt.plot(trainer_history_general.epoch, trainer_history_docu.loss, label='Documentation prompt', color='green')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig("results/loss_decoder.pdf")
plt.clf()


# plot loss for model without decoder input
trainer_history_general = pd.read_csv(M2_loss_path)
trainer_history_code = pd.read_csv(M4_loss_path)
trainer_history_docu = pd.read_csv(M6_loss_path)
plt.plot(trainer_history_general.epoch, trainer_history_general.loss, label='General prompt', color='blue')
plt.plot(trainer_history_general.epoch, trainer_history_code.loss, label='Coding-specific prompt', color='red')
plt.plot(trainer_history_general.epoch, trainer_history_docu.loss, label='Documentation prompt', color='green')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig("results/loss_no_decoder.pdf")
plt.clf()

# plot loss for model with encoder input
trainer_history_general = pd.read_csv(M7_loss_path)
trainer_history_code = pd.read_csv(M8_loss_path)
trainer_history_docu = pd.read_csv(M9_loss_path)
plt.plot(trainer_history_general.epoch, trainer_history_general.loss, label='General prompt', color='blue')
plt.plot(trainer_history_general.epoch, trainer_history_code.loss, label='Coding-specific prompt', color='red')
plt.plot(trainer_history_general.epoch, trainer_history_docu.loss, label='Documentation prompt', color='green')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig("results/loss_encoder.pdf")