import CyclicStyleTransfer as cst

similarity_weights = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [100]
image_pairs = [('house_photo.jpg', 'picasso.jpg'),
               ('neckarfront.jpg', 'vangogh.jpg'),
               ('opernhausZurich.jpg','SchonbrunnPalace.jpg'),
               ('landscape.jpg', 'steele.jpg')]

"""
for content_image_name, style_image_name in image_pairs:
    for similarity_weight in similarity_weights:
        cst.run_cyclic_style_transfer(content_image_name=content_image_name, style_image_name=style_image_name,
                                      content_weights=content_weights, style_weights=style_weights,
                                      similarity_type = 'content', similarity_weight=similarity_weight, add_index=True)
"""

cst.run_cyclic_style_transfer(content_image_name='guitar.jpg', style_image_name='cubism.jpg',
                                      content_weights=content_weights, style_weights=style_weights,
                                      similarity_type = 'lpsis', similarity_weight=100, add_index=True)