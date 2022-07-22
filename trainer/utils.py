import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def save_cmp_as_html(d_label, d_predict, d_masked,
                     well_trans, well_init, well_label,
                     well_predict1, well_predict2, output_path=None):

    def get_cpu(x):
        if type(x) is np.ndarray:
            return x
        else:
            return x.cpu().detach().numpy()

    d_label = get_cpu(d_label)
    d_masked = get_cpu(d_masked)
    d_predict = get_cpu(d_predict)
    well_label = get_cpu(well_label)
    well_init = get_cpu(well_init)
    well_predict1 = get_cpu(well_predict1)
    well_predict2 = get_cpu(well_predict2)

    batch_size, seq_len, _ = well_label.shape

    well_label = np.reshape(well_trans.inverse_transform(np.reshape(well_label, (-1, 3))), (batch_size, -1, 3))
    well_init = np.reshape(well_trans.inverse_transform(np.reshape(well_init, (-1, 3))), (batch_size, -1, 3))
    well_predict1 = np.reshape(well_trans.inverse_transform(np.reshape(well_predict1, (-1, 3))), (batch_size, -1, 3))
    well_predict2 = np.reshape(well_trans.inverse_transform(np.reshape(well_predict2, (-1, 3))), (batch_size, -1, 3))

    fig = make_subplots(rows=2, cols=3)

    y = list(range(seq_len))

    for k in range(3):
        fig.add_trace(
            go.Scatter(x=well_label[0, :, k], y=y, marker=dict(color="blue")),
            row=1, col=k + 1
        )

        fig.add_trace(
            go.Scatter(x=well_init[0, :, k], y=y, marker=dict(color="green")),
            row=1, col=k + 1
        )

        fig.add_trace(
            go.Scatter(x=well_predict1[0, :, k], y=y, marker=dict(color="red")),
            row=1, col=k + 1
        )

        fig.add_trace(
            go.Scatter(x=well_predict2[0, :, k], y=y, marker=dict(color="black")),
            row=1, col=k + 1
        )

        fig.add_trace(
            go.Scatter(x=d_label[0, :, k], y=y, marker=dict(color="blue")),
            row=2, col=k + 1
        )

        fig.add_trace(
            go.Scatter(x=d_masked[0, :, k], y=y, marker=dict(color="green")),
            row=2, col=k + 1
        )

        fig.add_trace(
            go.Scatter(x=d_predict[0, :, k], y=y, marker=dict(color="red")),
            row=2, col=k + 1
        )

    fig.write_html(f"{output_path}/cmp_real_predict.html")
