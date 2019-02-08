//KNN Classification

let video;
let features;
let knn;
let labelP;
let ready = false;

function setup() {
	createCanvas(450, 320);
	video = createCapture(VIDEO);
	video.size(450, 320);
	video.hide();
	features = ml5.featureExtractor("MobileNet", modelReady);
	knn = ml5.KNNClassifier();
	labelP = createP("Need training data");
	labelP.style('font-size', '32pt')

}

function goClassify() {
	const logits = features.infer(video);
	knn.classify(logits, function(error, result) {
		if(error) {
			console.error(error);
		} else {
			labelP.html(result.label);
			goClassify();
		}
	}); 
}



function keyPressed() {
	const logits = features.infer(video);
	if(key == 'l') {
		knn.addExample(logits, 'punch_left');
		console.log('punch_left');
	} else if(key == 'r') {
		knn.addExample(logits, 'punch_right');
		console.log('punch_right');
	}

	// console.log(logits.dataSync());
}

function modelReady() {
	console.log("model ready")
	// features.infer(video).print();2
}

function draw() {
	image(video, 0 , 0);
	if(!ready && knn.getNumLabels() > 0) {
		goClassify();
		ready = true;
	}
}