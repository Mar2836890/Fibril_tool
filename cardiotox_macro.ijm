run("8-bit");
run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");
run("Sharpen");
run("FFT");
setColor(0);
makeOval(811, 811, 425, 425);
fill();
makeOval(670, 670, 708, 708);
run("Make Inverse");
fill();
run("Inverse FFT");
a=getTitle();
selectImage(a);
run("Enhance Contrast...", "saturated=0.2 normalize");
setThreshold(80, 255);
run("Convert to Mask");
run("Analyze Particles...", "summarize");
//saveAs("Tiff", binaryDir + "/"+ list[i]);


//run("Auto Threshold", "method=MaxEntropy white");
