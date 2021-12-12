import os
from configargparse import ArgParser
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch.utils.data as Data
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import xlsxwriter


# def limit_mem():
#     """Limit TF GPU mem usage"""
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     tf.compat.v1.Session(config=config)

def generatedatafluxnet(datadir, dataname):
    # Open fluxnetdataset and create data generators
    # TODO: Flexible input data
    z = pd.read_csv(f'{datadir}{dataname}.csv')

    timedata = z.loc[:, ['TIMESTAMP']]
    # time variables:Year.
    yeartimedata = pd.DataFrame((timedata / 10e3), dtype='int')
    yeartimedata.rename(columns={'TIMESTAMP': 'yeartimedata'}, inplace=True)
    # time variables:month.
    monthtimedata = pd.DataFrame(pd.DataFrame((timedata / 10e1), dtype='int') % 1e2, dtype='int')
    monthtimedata.rename(columns={'TIMESTAMP': 'monthtimedata'}, inplace=True)
    # time variables:day.
    daytimedata = pd.DataFrame((pd.DataFrame((timedata / 1), dtype='int') % 1e2), dtype='int')
    daytimedata.rename(columns={'TIMESTAMP': 'daytimedata'}, inplace=True)
    # Surface soil temperature
    Tsdata = pd.DataFrame(z.loc[:, ['TS_F_MDS_1']])
    ####Longwave radiation
    Lr = pd.DataFrame(z.loc[:, ['LW_IN_F_MDS']])
    ####Shortwave radiation
    Sr = pd.DataFrame(z.loc[:, ['SW_IN_F_MDS']])
    ###Air temperature
    AT = pd.DataFrame(z.loc[:, ['TA_F_MDS']])
    ###Atmospheric pressure
    AP = pd.DataFrame(z.loc[:, ['PA_ERA']])
    ###Wind speed
    WS = pd.DataFrame(z.loc[:, ['WS_ERA']])
    ###Precipitation.
    Pdata = pd.DataFrame(z.loc[:, ['P_ERA']])
    P_oneweek_data = pd.DataFrame(z.loc[:, ['P_ERA']])
    ####计算1周降水积攒量
    for i in range(7, Pdata.size):
        P_oneweek = Pdata.values[i - 7:i]
        P_oneweek_data.values[i - 1] = np.sum(P_oneweek)
    # soil moisture  variables.
    SWCdata = pd.DataFrame(z.loc[:, ['SWC_F_MDS_1']])

    # concat all variables.
    # ds1 = pd.concat([monthtimedata,daytimedata,SWCdata,Lr,Sr,AT,AP,WS,Pdata,Tsdata],axis=1,join='inner')
    # ds1 = pd.concat([Lr,Sr,AP,AT,Tsdata],axis=1,join='inner')
    ds1 = pd.concat([Tsdata], axis=1, join='inner')
    # ds1 = pd.concat([AT,Lr,Tsdata, Pdata, AP,Lr,SWCdata], axis=1, join='inner')
    ds = Tsdata
    # soil moisture  variables normal.
    data = ds1.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(ds1)
    scaler1 = scaler1.fit(ds)
    # convert all variables to array.

    return data, scaler1, scaler


def LSTMDataGenerator(data, lead_time, batch_size, seq_length):
    # Currently, we have a big sequence of half hourly cases. We’ll convert it into smaller ones:
    train_xt, train_yt = create_sequences(data, seq_length, lead_time)
    train_xt = torch.from_numpy(train_xt).float().cuda()
    train_yt = torch.from_numpy(train_yt).float().cuda()
    train_zt = train_yt
    #
    return train_xt, train_yt, train_zt


def create_sequences(data, seq_length, lead_time):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - lead_time - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length + lead_time, data.shape[1] - 1]
        xs.append(x)
        ys.append(y)
    # return xs, ys
    return np.array(xs), np.array(ys)


class build_lstm(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_layers):
        super(build_lstm, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                                  num_layers=n_layers)

        self.linear = torch.nn.Linear(in_features=n_hidden, out_features=1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        output, (w, y) = self.lstm(x)
        output = self.dropout(output)
        output = output[:, -1, :]
        output = self.linear(output)
        return output


def train_lstm(model, lr, total_epoch, train_loader, data_valid_x, data_valid_y, model_save_fn):
    if (os.path.exists(model_save_fn)):
        checkpoint = torch.load(model_save_fn)
        model.load_state_dict(checkpoint['net'])
        model.cuda()
        print(model)
        print('model has existed')
    else:
        sumWriter = SummaryWriter('lstm_log')
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = torch.nn.SmoothL1Loss().cuda()
        # loss_func = torch.nn.MSELoss().cuda()
        global_step = 1
        # initialize the early_stopping object
        # early_stopping = EarlyStopping(patience=50, verbose=True)
        ########## training set##########
        for epoch in range(total_epoch):
            for step, (x, y) in enumerate(train_loader):
                output = model(x)
                train_loss = loss_func(output, y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                global_step = global_step + 1
            sumWriter.add_scalar("train_loss", train_loss.item() / global_step, global_step=global_step)

            # # #########设置模型为评估/测试模式#####################
            # # model.eval()
            # # 一般如果验证集不是很大的话，模型验证就不需要按批量进行了，但要注意输入参数的维度不能错
            valid_output = model(data_valid_x)
            # 注意这里的输入参数维度要符合要求，我这里为了简单，并未考虑这一点
            valid_loss = loss_func(valid_output, data_valid_y)
            # ########################################################
            print(epoch, 'tain_Loss:', train_loss.item(), 'validation_Loss:', valid_loss.item())
            # sumWriter.add_scalar("train_loss", train_loss.item() / global_step, global_step=global_step)
            # ########## validation set##########
        # state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        # torch.save(state, model_save_fn)
        # print(f'Saving model weights: {model_save_fn}')
        sumWriter.close()
        return model


def create_predictions(model, data_test_x, scaler):
    preds = model(data_test_x)
    # Unnormalize
    preds = preds.cpu()
    preds = preds.detach().numpy()
    # Unnormalize
    preds = scaler.inverse_transform(preds)
    return preds


def compute_rmse_r2(data_testd_y, pred, modelname):
    # 计算MSE
    rmse = np.sqrt(mean_squared_error(data_testd_y, pred))
    mae = mean_absolute_error(data_testd_y, pred)
    r2 = r2_score(data_testd_y, pred)
    bias = np.mean(pred - data_testd_y)
    print(f"均方误差(RMSE)：{rmse}")
    print(f"均方误差(MAE)：{mae}")
    print(f"测试集R^2：{r2}")
    print(f"测试集bias：{bias}")
    plt.figure(figsize=(12, 8))
    plt.plot(data_testd_y, label='True soil moisture')
    plt.plot(pred, label='Soil moisture prediction ')
    plt.legend(loc='best')
    plt.text(10, 10, 'R2=%.3f' % r2, fontdict={'size': 20, 'color': 'b'}
             , verticalalignment="bottom", horizontalalignment="left")
    plt.title(modelname)
    plt.show()


def Resultsave(y_test, filename, dir):
    county = 0
    countz = 1
    f = os.path.join(dir, filename)
    xl = xlsxwriter.Workbook(f)
    sheet = xl.add_worksheet()
    for i in range(y_test.shape[0]):
        sheet.write(i, county, y_test[i][0])
        sheet.write(i, countz, y_test[i][1])
    xl.close()
    print("write_excel over")


def main(datadir, dataname, hidden_size, num_layers, lr, total_epoch, batch_size, lead_time,
         seq_length, model_save_fn, modelname):
    # TODO: Flexible input data
    # Open fluxnetdataset and create data generators
    data, scaler, scaler1 = generatedatafluxnet(datadir, dataname)

    # TODO: Normalization
    # scaler1 = MinMaxScaler(feature_range=(0, 1))
    # scaler1 = scaler1.fit(data)
    data = scaler1.transform(data)
    # TODO: Generate the tensor for lstm model

    [data_x, data_y, data_z] = LSTMDataGenerator(data, lead_time, batch_size, seq_length)

    # TODO: Flexible valid split
    data_train_x = data_x[:int((data_x.shape[0]) - 400)]
    data_train_y = data_y[:int(data_x.shape[0] - 400)]

    train_data = Data.TensorDataset(data_train_x, data_train_y)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    data_valid_x = data_x[int(data_x.shape[0] - 400):int(data_x.shape[0] - 365)]
    data_valid_y = data_y[int(data_x.shape[0] - 400):int(data_x.shape[0] - 365)]
    data_test_x = data_x[int(data_x.shape[0] - 365):int(1.0 * data_x.shape[0])]
    data_testd_z = data_z[int(data_x.shape[0] - 365):int(1.0 * data_x.shape[0])]

    # TODO: Flexible input shapes and optimizer
    model = build_lstm(n_features=data_x.shape[2], n_hidden=hidden_size, n_layers=num_layers)
    # TODO: Trian LSTM based on the training and validation sets
    model = train_lstm(model, lr, total_epoch, train_loader, data_valid_x, data_valid_y, model_save_fn)
    # TODO: Create predictions based on the test sets
    pred = create_predictions(model, data_test_x, scaler)
    # TODO: Computer score of R2 and RMSE

    data_testd_z = data_testd_z.reshape(-1, 1)
    data_testd_z = data_testd_z.cpu()
    data_testd_z = data_testd_z.detach().numpy()
    # Unnormalize
    data_testd_z = scaler.inverse_transform(data_testd_z)
    compute_rmse_r2(data_testd_z, pred, modelname)

    # TODO: Computer score of R2 and RMSE

    dir = r"./results"
    if not os.path.exists(dir):
        os.mkdir(dir)

    resultname = modelname + dataname + '.csv'
    testresult = np.hstack((data_testd_z, pred))
    Resultsave(testresult, resultname, dir)


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--datadir', type=str, default='./data/', help='Path to data')
    p.add_argument('--dataname', type=str, default='FLX_US-Goo', help='Location to data')
    p.add_argument('--hidden_size', type=int, default=16, help='hidden sizes for lstm model')
    p.add_argument('--num_layers', type=int, default=2, help='layer unmber for lstm model')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--total_epoch', type=int, default=100, help='total epochs for training the model')
    p.add_argument('--batch_size', type=int, default=16, help='batch_size')
    p.add_argument('--lead_time', type=int, default=1, help='Forecast lead time')
    p.add_argument('--seq_length', type=int, default=10, help='input timesteps for lstm model')
    p.add_argument('--model_save_fn', type=str, default='./saved_models/lstm_1d.h5', help='Path to save model')
    p.add_argument('--modelname', type=str, default='lstm-3days-daily-prediciton', help='name for prediction model')
    args = p.parse_args()

    main(
        datadir=args.datadir,
        dataname=args.dataname,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lr=args.lr,
        total_epoch=args.total_epoch,
        batch_size=args.batch_size,
        lead_time=args.lead_time,
        seq_length=args.seq_length,
        model_save_fn=args.model_save_fn,
        modelname=args.modelname,
    )
