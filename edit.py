from AutoVideoEdit import AVE
vid1 = AVE()
vid1.addVideo('./test_vids/random_cat.mp4')
vid1.newQuery('petting a cat')
vid1.addVideo('./test_vids/baby_penguin.mp4')
vid1.newQuery('small white penguin')
vid1.newQuery('penguin underwater')
vid1.compile_vid()