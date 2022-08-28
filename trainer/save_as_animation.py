import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def save_as_animation(epoch, iteration, d_label, d_predict, d_masked,
                      well_trans, well_init, well_label,
                      well_predict1, well_predict2,
                      output_path=None,
                      frames=None, index=0, fig=None):

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

    y = list(range(seq_len))

    def get_traces():
        all_traces = []
        animation_data = []
        for k in range(3):
            traces = [
                go.Scatter(x=well_label[0, :, k], y=y, marker=dict(color="blue"), showlegend=False),
                go.Scatter(x=well_init[0, :, k], y=y, marker=dict(color="green"), showlegend=False),
                go.Scatter(x=well_predict1[0, :, k], y=y, marker=dict(color="red"), showlegend=False),
                go.Scatter(x=well_predict2[0, :, k], y=y, marker=dict(color="black"), showlegend=False),

                go.Scatter(x=d_masked[0, :, k * 10], y=y, marker=dict(color="green"), showlegend=False),
                go.Scatter(x=d_label[0, :, k*10], y=y, marker=dict(color="blue"), showlegend=False),
                go.Scatter(x=d_predict[0, :, k*10], y=y, marker=dict(color="red"), showlegend=False),
            ]
            # traces = [
            #     go.Scatter(x=np.random.randn(seq_len), y=y, marker=dict(color="blue"), showlegend=False),
            #     go.Scatter(x=np.random.randn(seq_len), y=y, marker=dict(color="green"), showlegend=False),
            #     go.Scatter(x=np.random.randn(seq_len), y=y, marker=dict(color="red"), showlegend=False),
            #     go.Scatter(x=np.random.randn(seq_len), y=y, marker=dict(color="black"), showlegend=False),
            #
            #     go.Scatter(x=np.random.randn(seq_len), y=y, marker=dict(color="blue"), showlegend=False),
            #     go.Scatter(x=np.random.randn(seq_len), y=y, marker=dict(color="green"), showlegend=False),
            #     go.Scatter(x=np.random.randn(seq_len), y=y, marker=dict(color="red"), showlegend=False),
            # ]

            all_traces.append(traces)
            for trace in traces:
                animation_data.append(trace)

        traces = list(range(len(animation_data)))
        return all_traces, animation_data, traces

    all_traces, animation_data, traces = get_traces()

    if fig is None:
        names = ["VP", "VS", r"Rho",
                 r"Angle (0)", r"Angle (17.5)", r"Angle (35)"]
        fig = make_subplots(rows=2, cols=3, subplot_titles=names, vertical_spacing=0.08)

        for k in range(3):
            traces = all_traces[k]
            for j in range(7):
                if j < 4:
                    fig.add_trace(traces[j], row=1, col=k + 1)
                else:
                    fig.add_trace(traces[j], row=2, col=k + 1)

        # Update xaxis properties
        x_axes = [["Velocity (m/s)", "Velocity (m/s)", "Density (g/cm^3)"],
                   ["Amplitude", "Amplitude", "Amplitude"]]
        for row in range(2):
            for col in range(3):
                fig.update_xaxes(title_text=x_axes[row][col], row=row+1, col=col+1)
                fig.update_yaxes(title_text="Samples", row=row+1, col=col+1)

    # layout = go.Layout(
    #     annotations=[
    #         go.layout.Annotation(
    #             x=1.25,
    #             y=1.25,
    #             text=f"Epoch={epoch}, iteration={iteration}",
    #         )
    #     ],
    # )
    frame = go.Frame(data=animation_data, traces=traces)

    if not frames:
        frames = [frame]
    else:
        frames.append(frame)

    fig.frames = frames
    button = dict(
        label='Play',
        method='animate',
        args=[None, dict(frame=dict(duration=100, redraw=True),
                         transition=dict(duration=50),
                         fromcurrent=False,
                         mode='immediate')])
    fig.update_layout(updatemenus=[dict(type='buttons',
                                        showactive=True,
                                        y=0.55,
                                        x=1.05,
                                        xanchor='left',
                                        yanchor='bottom',
                                        buttons=[button])
                                   ],
                      width=1500, height=1300)

    # fig.update_layout(yaxis2_range=[0, 5.5], yaxis2_autorange=False)
    # fig.show()

    fig.write_html(f"{output_path}/animation_{index}.html")

    return frames, fig
