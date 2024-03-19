@echo off
setlocal enabledelayedexpansion

set PRE_SEQ_LEN=128
set LR=2e-2
set NUM_GPUS=1
set MAX_SEQ_LEN=2048
set DEV_BATCH_SIZE=1
set GRAD_ACCUMULARION_STEPS=16
set MAX_STEP=1000
set SAVE_INTERVAL=500

for /f "delims=" %%a in ('date /t') do set DATESTR=%%a
for /f "delims=" %%a in ('time /t') do set TIMESTR=%%a
set DATESTR=%DATESTR:-=%
set TIMESTR=%TIMESTR::=%
set DATESTR=%DATESTR: =%
set TIMESTR=%TIMESTR: =%
set RUN_NAME=tool_alpaca_pt

set BASE_MODEL_PATH=.././chatglm3-6b
set DATASET_PATH=scripts/formatted_data/tool_alpaca.jsonl
set OUTPUT_DIR=output/%RUN_NAME%-%DATESTR%-%TIMESTR%-%PRE_SEQ_LEN%-%LR%

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

torchrun --standalone --nnodes=1 --nproc_per_node=%NUM_GPUS% finetune.py ^
    --train_format multi-turn ^
    --train_file %DATASET_PATH% ^
    --max_seq_length %MAX_SEQ_LEN% ^
    --preprocessing_num_workers 1 ^
    --model_name_or_path %BASE_MODEL_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --per_device_train_batch_size %DEV_BATCH_SIZE% ^
    --gradient_accumulation_steps %GRAD_ACCUMULARION_STEPS% ^
    --max_steps %MAX_STEP% ^
    --logging_steps 1 ^
    --save_steps %SAVE_INTERVAL% ^
    --learning_rate %LR% ^
    --pre_seq_len %PRE_SEQ_LEN% > "%OUTPUT_DIR%\train.log" 2>&1

endlocal
