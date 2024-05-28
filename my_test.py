from mmdet.apis import DetInferencer

# Initialize the DetInferencer
model_path = ""
weight_path = ""
inferencer = DetInferencer(model=model_path, weights=weight_path, device="cuda:1")

# Perform inference
inferencer("demo_image/", out_dir="my_test", pred_score_thr=0.5)