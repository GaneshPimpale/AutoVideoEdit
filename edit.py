from AutoVideoEdit import AVE
vid1 = AVE()
vid1.addVideo('./test_vids/random_cat.mp4')
vid1.newQuery('cat petting a')
vid1.addVideo('./test_vids/baby_penguin.mp4')
vid1.newQuery('penguin white small')
vid1.newQuery('is penguin that underwater')
vid1.compile_vid()