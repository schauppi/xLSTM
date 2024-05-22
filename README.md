Paper implementation of the memory cells sLSTM and mLSTM as well as the xLSTM architecture presented in [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517).

![Figure 1](images/fig_1.png)

# Initial Results

Results on an sine wave with 5 hidden units and 500 epochs.

## sLSTM

![Figure 2](images/sLSTMCell_10.png)

## mLSTM

![Figure 3](images/mLSTMCell_10.png)

# To Do

- [X] Check implementation of mLSTM - seems somewhat off
- [ ] Implement xLSTM - stack Cells together
