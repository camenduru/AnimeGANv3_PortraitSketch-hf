#!/usr/bin/env python

from __future__ import annotations

import argparse
import pathlib

import gradio as gr

from model import Model

DESCRIPTION = '''# AnimeGANv3 Portrait Sketch

<img id="overview" alt="overview" src="https://raw.githubusercontent.com/TachibanaYoshino/AnimeGANv3/master/results/face2portrait_sketch.jpg" />

This is an unofficial demo app for [AnimeGANv3 Portrait Sketch](https://github.com/TachibanaYoshino/AnimeGANv3).
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.animeganv3_portrait_sketch" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def main():
    args = parse_args()
    model = Model(device=args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
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

        gr.Markdown(FOOTER)

        run_button.click(fn=model.run, inputs=input_image, outputs=result)

        example_images.click(fn=set_example_image,
                             inputs=example_images,
                             outputs=example_images.components)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
