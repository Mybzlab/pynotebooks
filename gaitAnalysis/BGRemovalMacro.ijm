run("Z Project...", "projection=Median");
imageCalculator("Subtract stack", 1,2);
selectImage(1);
run("Export Movie Using FFmpeg...", "last_slice=400 frame_rate=60 format=avi encoder=[by format] custom_encoder=[] add_timestamp save=[C:/Users/martv/Documents/robot video/temp/"+getTitle()+".avi]");
//waitForUser;
run("Close All");