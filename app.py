#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr

from model import Model

DESCRIPTION = '''# AnimeGANv3 Portrait Sketch

<img id="overview" alt="overview" src="https://github.com/TachibanaYoshino/AnimeGANv3/raw/0c8fe412e451131f8998572e8d48b1bff1952611/results/face2portrait_sketch.jpg" />

This is an unofficial demo app for [AnimeGANv3 Portrait Sketch](https://github.com/TachibanaYoshino/AnimeGANv3).
'''


def set_example_image(example: list) -> dict:
    return gr.update(value=example[0])


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
        example_images = gr.Dataset(components=[input_image],
                                    samples=[[path.as_posix()]
                                             for path in paths])

    run_button.click(fn=model.run, inputs=input_image, outputs=result)

    example_images.click(fn=set_example_image,
                         inputs=example_images,
                         outputs=example_images.components)

demo.queue().launch(show_api=False)
