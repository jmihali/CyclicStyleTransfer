import CyclicStyleTransfer as cst
import NeuralStyleTransfer as nst
from CyclicStyleTransfer import MSELoss_images, content_loss

similarity_type_weights = [
    {'mse' : 10,
    'content': 0},
    {'mse': 1e5,
     'content': 0},
    {'mse' : 0,
    'content': 10},
    {'mse': 0,
     'content': 1e5}
]

style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
content_weights = [100]

image_pairs = [
    ('portrait.jpg', 'abstract.jpg'),
    ('lion_photo.jpg', 'picasso.jpg'),
    ('neckarfront.jpg', 'vangogh.jpg'),
]

out_dir = "Images/Results"
cnt=1
for similarity_type_weight in similarity_type_weights:
    for content_image_name, style_image_name in image_pairs:
        stylized_image_path = nst.run_neural_style_transfer(content_image_name=content_image_name, style_image_name=style_image_name,
                                      content_weights=content_weights, style_weights=style_weights,
                                      add_index=True, output_dir=out_dir)

        w1, w2, w3 = stylized_image_path.split(sep='/')
        stylized_image_name = w2 + '/' + w3

        reversed_image_path = nst.run_neural_style_transfer(content_image_name=stylized_image_name, style_image_name=content_image_name,
                                      content_weights=content_weights, style_weights=style_weights,
                                      add_index=True, output_dir=out_dir)

        print("MSE loss between content and reversed image:", MSELoss_images(stylized_image_path, reversed_image_path).item())
        print("Content loss between content and reversed image:", content_loss(stylized_image_path, reversed_image_path).item())
        print("=" * 15)
        print("\n")


        cst.run_cyclic_style_transfer(content_image_name=content_image_name, style_image_name=style_image_name,
                                      content_weights=content_weights, style_weights=style_weights,
                                      similarity_type_weight=similarity_type_weight, add_index=True, output_dir=out_dir)
        stylized_image_path = "Images/Results/cst_stylized_image%d.jpg"%cnt
        reversed_image_path = "Images/Results/cst_reversed_image%d.jpg"%cnt
        print("MSE loss between content and reversed image:", MSELoss_images(stylized_image_path, reversed_image_path).item())
        print("Content loss between content and reversed image:", content_loss(stylized_image_path, reversed_image_path).item())
        print("="*15)

        print("\n\n\n")
        cnt += 1