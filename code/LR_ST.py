import os
from configargparse import ArgParser
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


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
    # ds1 = pd.concat([Lr, Sr, AP, AT, Tsdata], axis=1, join='inner')
    ds1 = pd.concat([SWCdata], axis=1, join='inner')
    # ds1 = pd.concat([AT,Lr,Tsdata, Pdata, AP,Lr,SWCdata], axis=1, join='inner')
    ds = SWCdata
    # soil moisture  variables normal.
    data = ds1.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(ds1)
    scaler1 = scaler1.fit(ds)
    # convert all variables to array.

    return data, scaler1, scaler


def SVRDataGenerator(data, lead_time, seq_length):
    # Currently, we have a big sequence of half hourly cases. We’ll convert it into smaller ones:
    train_xt, train_yt = create_sequences(data, seq_length, lead_time)
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


def compute_rmse_r2(data_testd_y, pred, modelname):
    # 计算MSE
    rmse = np.sqrt(mean_squared_error(data_testd_y, pred))
    mae = mean_absolute_error(data_testd_y, pred)
    r2 = r2_score(data_testd_y, pred)
    print(f"均方误差(RMSE)：{rmse}")
    print(f"均方误差(MAE)：{mae}")
    print(f"测试集R^2：{r2}")

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


def main(datadir, dataname, lead_time,
         seq_length, modelname):
    # TODO: Flexible input data
    # Open fluxnetdataset and create data generators
    data, scaler, scaler1 = generatedatafluxnet(datadir, dataname)

    # TODO: Normalization
    # scaler1 = MinMaxScaler(feature_range=(0, 1))
    # scaler1 = scaler1.fit(data)
    data = scaler1.transform(data)
    # TODO: Generate the tensor for lstm model

    [data_x, data_y, data_z] = SVRDataGenerator(data, lead_time, seq_length)

    # TODO: Flexible valid split
    data_train_x = data_x[:int((data_x.shape[0]) - 400)]
    data_train_y = data_y[:int(data_x.shape[0] - 400)]
    data_test_x = data_x[int(data_x.shape[0] - 365):int(1.0 * data_x.shape[0])]
    data_testd_z = data_z[int(data_x.shape[0] - 365):int(1.0 * data_x.shape[0])]

    # TODO: Flexible input shapes and optimizer

    svr_input = data_train_x.reshape((data_train_x.shape[0], data_train_x.shape[1] * data_train_x.shape[2]))
    data_train_y = data_train_y.reshape((data_train_y.shape[0], 1))
    # TODO: Trian SVR based on the training and validation sets
    # parametersGrid = {"max_iter": [1, 100, 200],
    #                   "alpha": [0.1, 1, 10, 100],
    #                   "l1_ratio": [0, 0.5, 1]}
    # parametersGrid = {"max_iter": [1, 100, 200],
    #                   "learning_rate": [0.01, 0.02, 0.03]}
    # eNet = ElasticNet()
    eNet = LinearRegression()
    # grid = GridSearchCV(eNet, parametersGrid, cv=10)
    grid = linear_model.LinearRegression(normalize=False)
    grid.fit(svr_input, data_train_y)
    # TODO: Create predictions based on the test sets
    svr_input_test = data_test_x.reshape((data_test_x.shape[0], data_test_x.shape[1] * data_test_x.shape[2]))
    pred = grid.predict(svr_input_test)
    pred = pred.reshape(-1, 1)
    pred = scaler.inverse_transform(pred)

    # pred= preds[:]
    # TODO: Computer score of R2 and RMSE

    data_testd_z = data_testd_z.reshape(-1, 1)
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
    p.add_argument('--dataname', type=str, default='FLX_FR-LBr', help='Location to data')
    p.add_argument('--lead_time', type=int, default=1, help='Forecast lead time')
    p.add_argument('--seq_length', type=int, default=10, help='input timesteps for lstm model')
    p.add_argument('--modelname', type=str, default='ENET-5days-daily-prediciton', help='name for prediction model')
    args = p.parse_args()

    main(
        datadir=args.datadir,
        dataname=args.dataname,
        lead_time=args.lead_time,
        seq_length=args.seq_length,
        modelname=args.modelname,
    )
