# Supervision
[![version](https://badge.fury.io/py/supervision.svg)](https://badge.fury.io/py/supervision)
[![downloads](https://img.shields.io/pypi/dm/supervision)](https://pypistats.org/packages/supervision)
[![snyk](https://snyk.io/advisor/python/supervision/badge.svg)](https://snyk.io/advisor/python/supervision)
[![license](https://img.shields.io/pypi/l/supervision)](https://github.com/roboflow/supervision/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/supervision)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/supervision/blob/main/demo.ipynb)
[![gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Roboflow/Annotators)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)
[![built-with-material-for-mkdocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

## 👋 hello

**Supervision: The missing toolkit for computer vision developers**

Supervision is a set of easy-to-use utilities that streamline your computer vision workflow. It abstracts away complex boilerplate code, letting you focus on building powerful applications instead of reinventing the wheel.

### What Makes Supervision Different?

- **Truly Model-Agnostic**: Works seamlessly with all popular CV models (YOLO, MMDetection, HuggingFace, etc.)
- **Modular Design**: Use only what you need - detection, tracking, visualization, dataset handling
- **Production-Ready**: Optimized for real-world applications with high performance requirements
- **Extensive Documentation**: Comprehensive guides, cookbooks, and API references

Whether you're prototyping a concept or deploying to production, Supervision provides the tools to get there faster.

## 💻 installation options

### Pip (Recommended)
```bash
pip install supervision
```

### With metrics support
```bash
pip install "supervision[metrics]"
```

### Using Poetry
```bash
poetry add supervision
# With metrics support
poetry add "supervision[metrics]"
```

### From source (for developers)
```bash
git clone https://github.com/roboflow/supervision.git
cd supervision
pip install -e .
```

### Version compatibility
- Works with **Python 3.8-3.13**
- Thoroughly tested across Linux, macOS, and Windows

For more detailed installation information, visit our [installation guide](https://roboflow.github.io/supervision/).

## 🔐 Environment Variables

Create a `.env` file to store sensitive configuration:

```bash
ROBOFLOW_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

Then load them in your code:
```python
from dotenv import load_dotenv

load_dotenv()  # Load before other imports
# Now use os.getenv() to access values
```


## 🔥 quickstart

### models

Supervision was designed to be model agnostic. Just plug in any classification, detection, or segmentation model. For your convenience, we have created [connectors](https://supervision.roboflow.com/latest/detection/core/#detections) for the most popular libraries like Ultralytics, Transformers, or MMDetection.

```python
import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread(...)
model = YOLO("yolov8s.pt")
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

len(detections)
# 5
```

<details>
<summary>👉 more model connectors</summary>

- inference

  Running with [Inference](https://github.com/roboflow/inference) requires a [Roboflow API KEY](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

  ```python
  import cv2
  import supervision as sv
  from inference import get_model

  image = cv2.imread(...)
  model = get_model(model_id="yolov8s-640", api_key=<ROBOFLOW API KEY>)
  result = model.infer(image)[0]
  detections = sv.Detections.from_inference(result)

  len(detections)
  # 5
  ```

</details>

### annotators

Supervision offers a wide range of highly customizable [annotators](https://supervision.roboflow.com/latest/detection/annotators/), allowing you to compose the perfect visualization for your use case.

```python
import cv2
import supervision as sv

image = cv2.imread(...)
detections = sv.Detections(...)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(
  scene=image.copy(),
  detections=detections)
```

https://github.com/roboflow/supervision/assets/26109316/691e219c-0565-4403-9218-ab5644f39bce

### datasets

Supervision provides a set of [utils](https://supervision.roboflow.com/latest/datasets/core/) that allow you to load, split, merge, and save datasets in one of the supported formats.

```python
import supervision as sv
from roboflow import Roboflow

project = Roboflow().workspace(<WORKSPACE_ID>).project(<PROJECT_ID>)
dataset = project.version(<PROJECT_VERSION>).download("coco")

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
)

path, image, annotation = ds[0]
    # loads image on demand

for path, image, annotation in ds:
    # loads image on demand
```

<details>
<summary>👉 more dataset utils</summary>

- load

  ```python
  dataset = sv.DetectionDataset.from_yolo(
      images_directory_path=...,
      annotations_directory_path=...,
      data_yaml_path=...
  )

  dataset = sv.DetectionDataset.from_pascal_voc(
      images_directory_path=...,
      annotations_directory_path=...
  )

  dataset = sv.DetectionDataset.from_coco(
      images_directory_path=...,
      annotations_path=...
  )
  ```

- split

  ```python
  train_dataset, test_dataset = dataset.split(split_ratio=0.7)
  test_dataset, valid_dataset = test_dataset.split(split_ratio=0.5)

  len(train_dataset), len(test_dataset), len(valid_dataset)
  # (700, 150, 150)
  ```

- merge

  ```python
  ds_1 = sv.DetectionDataset(...)
  len(ds_1)
  # 100
  ds_1.classes
  # ['dog', 'person']

  ds_2 = sv.DetectionDataset(...)
  len(ds_2)
  # 200
  ds_2.classes
  # ['cat']

  ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
  len(ds_merged)
  # 300
  ds_merged.classes
  # ['cat', 'dog', 'person']
  ```

- save

  ```python
  dataset.as_yolo(
      images_directory_path=...,
      annotations_directory_path=...,
      data_yaml_path=...
  )

  dataset.as_pascal_voc(
      images_directory_path=...,
      annotations_directory_path=...
  )

  dataset.as_coco(
      images_directory_path=...,
      annotations_path=...
  )
  ```

- convert

  ```python
  sv.DetectionDataset.from_yolo(
      images_directory_path=...,
      annotations_directory_path=...,
      data_yaml_path=...
  ).as_pascal_voc(
      images_directory_path=...,
      annotations_directory_path=...
  )
  ```

</details>

### tracking

Track objects across video frames with ByteTrack integration:

```python
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=20)

video_info = sv.VideoInfo.from_video_path("video.mp4")
frames_generator = sv.get_video_frames_generator("video.mp4")

with sv.VideoSink("output.mp4", video_info) as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        
        # Create labels with tracking IDs
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        # Annotate frame with both boxes and traces
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )
        
        sink.write_frame(annotated_frame)
```

https://github.com/roboflow/supervision/assets/26109316/3ac6982f-4943-4108-9b7f-51787ef1a69f

### speed estimation example

Use perspective transformation to estimate object speed from a video:

```python
# From examples/speed_estimation/ultralytics_example.py
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv

# Define perspective transform points
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH, TARGET_HEIGHT = 25, 250
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# Initialize components
video_info = sv.VideoInfo.from_video_path("vehicles.mp4")
model = YOLO("yolov8x.pt")
byte_track = sv.ByteTrack(frame_rate=video_info.fps)
polygon_zone = sv.PolygonZone(polygon=SOURCE)

# Create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)
trace_annotator = sv.TraceAnnotator(trace_length=video_info.fps * 2)

# Process video
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
```

Check the complete example in the [examples directory](https://github.com/roboflow/supervision/tree/main/examples/speed_estimation).

### metrics

Evaluate model performance with comprehensive metrics:

```python
import supervision as sv
from ultralytics import YOLO

# Load model and dataset
model = YOLO("yolov8n.pt")
dataset = sv.DetectionDataset.from_yolo(...)

# Create detection callback
def callback(image):
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)

# Run benchmarks
precision = sv.metrics.Precision(metric_target=sv.metrics.MetricTarget.BOXES)
recall = sv.metrics.Recall()
f1_score = sv.metrics.F1Score()
mean_ap = sv.metrics.MeanAveragePrecision()

# Process dataset and update metrics
for _, image, ground_truth in dataset:
    predictions = callback(image)
    precision.update(predictions, ground_truth)
    recall.update(predictions, ground_truth)
    f1_score.update(predictions, ground_truth)
    mean_ap.update(predictions, ground_truth)

# Get results
precision_result = precision.compute()
recall_result = recall.compute()
f1_score_result = f1_score.compute()
map_result = mean_ap.compute()

print(f"Precision@50: {precision_result.precision_at_50:.4f}")
print(f"Recall@50: {recall_result.recall_at_50:.4f}")
print(f"F1@50: {f1_score_result.f1_50:.4f}")
print(f"mAP@50-95: {map_result.map50_95:.4f}")

# Visualize results
precision_result.plot()
map_result.plot()
```

## 💜 built with supervision

https://user-images.githubusercontent.com/26109316/207858600-ee862b22-0353-440b-ad85-caa0c4777904.mp4

https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900

https://github.com/roboflow/supervision/assets/26109316/3ac6982f-4943-4108-9b7f-51787ef1a69f

## 📝 examples

Explore our real-world examples:

- **[Tracking](./examples/tracking)**: Object tracking with ByteTrack
- **[Count People in Zone](./examples/count_people_in_zone)**: Count objects within defined areas
- **[Traffic Analysis](./examples/traffic_analysis)**: Analyze vehicle movement patterns
- **[Speed Estimation](./examples/speed_estimation)**: Calculate object speeds using perspective transforms
- **[Time in Zone](./examples/time_in_zone)**: Measure how long objects remain in defined areas
- **[Heatmap and Track](./examples/heatmap_and_track)**: Visualize movement patterns with heatmaps

## 📚 documentation & resources

- **[Official Documentation](https://supervision.roboflow.com/)**: Comprehensive guides, API references, and examples
- **[Colab Notebook](https://colab.research.google.com/github/roboflow/supervision/blob/main/demo.ipynb)**: Interactive demo you can run in your browser
- **[Cheatsheet](https://roboflow.github.io/cheatsheet-supervision/)**: Quick reference for common operations
- **[Discord Community](https://discord.gg/GbfgXGJ8Bk)**: Join us for discussions, questions, and collaboration

## 🏆 contribution

We welcome contributions of all sizes! Here's how to get involved:

1. **Start with Issues**: Look for [issues labeled "good first issue"](https://github.com/roboflow/supervision/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
2. **Read the Guide**: Our [CONTRIBUTING.md](https://github.com/roboflow/supervision/blob/main/CONTRIBUTING.md) has detailed instructions
3. **Setup Dev Environment**: 
   ```bash
   git clone https://github.com/roboflow/supervision.git
   cd supervision
   pip install -e ".[dev]"
   pre-commit install
   ```
4. **Run Tests**: Ensure your changes pass all tests with `pytest`

Want to contribute but not sure where to start? Join our [Discord](https://discord.gg/GbfgXGJ8Bk) and we'll help you find a suitable project!

## 📊 citation

If you use Supervision in your research, please cite:

```bibtex
@misc{supervision2023,
  author = {Roboflow},
  title = {Supervision: A set of utilities for computer vision},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/roboflow/supervision}}
}
```
