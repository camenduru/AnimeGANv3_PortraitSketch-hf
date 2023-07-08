#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr

from model import Model

DESCRIPTION = '''# [AnimeGANv3 Portrait Sketch](https://github.com/TachibanaYoshino/AnimeGANv3)

<img id="overview" alt="overview" src="https://github.com/TachibanaYoshino/AnimeGANv3/raw/0c8fe412e451131f8998572e8d48b1bff1952611/results/face2portrait_sketch.jpg" />
'''

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(label='Input Image', type='numpy')
            with gr.Row():
                run_button = gr.Button('Run')
        with gr.Column():
            with gr.Row():
                result = gr.Image(label='Result',
                                  type='numpy',
                                  elem_id='result')
    with gr.Row():
        paths = sorted(pathlib.Path('images').glob('*.jpg'))
        gr.Examples(examples=[[path.as_posix()] for path in paths],
                    inputs=input_image)
    run_button.click(fn=model.run,
                     inputs=input_image,
                     outputs=result,
                     api_name='run')
demo.queue(max_size=20).launch()
