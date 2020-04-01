import CyclicStyleTransfer as cst

similarity_weight = 1e4
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [100]
image_pairs = [('house_photo.jpg', 'house_painting.jpg'),
               ('joan.jpg', 'picasso.jpg'),
               ('opernhausZurich.jpg','SchonbrunnPalace.jpg'),
               ('neckarfront.jpg', 'vangogh.jpg'),
               ('landscape.jpg', 'steele.jpg'),
               ('selena.jpg', 'monalisa.jpg')]

for content_image_name, style_image_name in image_pairs:
    cst.run_cyclic_style_transfer(content_image_name=content_image_name, style_image_name=style_image_name,
                                  content_weights=content_weights, style_weights=style_weights,
                                  similarity_type = 'content', similarity_weight=similarity_weight, add_index=True)