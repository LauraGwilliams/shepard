scenario = "shep";
write_codes = true;         
no_logfile = false;
pulse_width = 5;

begin;

#change to be 300ms tone 200ms off
trial {
   start_delay = 0;
   trial_duration = 480;
   stimulus_event {
		sound {
		wavefile {filename = ""; preload = false;} wav;
		} sound;
		code = 0;
      port_code = 1;
   } stim_event;
} tone_pres;

trial {
	trial_type = fixed;
	trial_duration = 500;
} silence;

begin_pcl;
include "Generic_subroutines.pcl";

#wav, condition, key, hz
int nFields = 4;
array<string> scaleA[8][nFields];
array<string> scaleC[8][nFields];
array<string> scaleEb[8][nFields];
array<string> randA[7][nFields];
array<string> randC[7][nFields];
array<string> randEb[7][nFields];
array<string> upScaleBlocks[3][0][0];
array<string> downScaleBlocks[3][0][0];
array<string> upCircBlocks[3][0][0];
array<string> downCircBlocks[3][0][0];
array<string> randomBlocks[3][0][0];
array<string> blocks[0][105][nFields];
array<string> outData[1665][9];
int trial_ct = 1;

array<string> trialInfo[4];
string blockType;
array<string> blockOrder[] = {"upScaleA", "upScaleC", "upScaleEb", "downScaleA", "downScaleC", "downScaleEb", "upCircA", "downCircA", "upCircC", "downCircC", "upCircEb", "downCircEb", "randomA", "randomC", "randomEb"};
blockOrder.shuffle();

input_file infile1 = new input_file;
input_file infile2 = new input_file;
input_file infile3 = new input_file;
infile1.open("scaleAShep.txt");
infile2.open("scaleCShep.txt");
infile3.open("scaleEbShep.txt");
input_file infile4 = new input_file;
input_file infile5 = new input_file;
input_file infile6 = new input_file;
infile4.open("scaleAShep.txt");
infile5.open("scaleCShep.txt");
infile6.open("scaleEbShep.txt");

loop
	int row = 1
until
	row > 8
begin
	loop
		int col = 1
	until
		col > nFields
	begin
		scaleA[row][col] = infile1.get_string();
		scaleC[row][col] = infile2.get_string();
		scaleEb[row][col] = infile3.get_string();
		col = col + 1;
	end;
	row = row + 1;
end;

loop
	int row = 1
until
	row > 7
begin
	loop
		int col = 1
	until
		col > nFields
	begin
		randA[row][col] = infile4.get_string();
		randC[row][col] = infile5.get_string();
		randEb[row][col] = infile6.get_string();
		col = col + 1;
	end;
	row = row + 1;
end;


loop
	int i = 1
until
	i > 8
begin
	upScaleBlocks[1].add(scaleA[i]);
	upScaleBlocks[2].add(scaleC[i]);
	upScaleBlocks[3].add(scaleEb[i]);
	downScaleBlocks[1].add(scaleA[abs(i-9)]);
	downScaleBlocks[2].add(scaleC[abs(i-9)]);
	downScaleBlocks[3].add(scaleEb[abs(i-9)]);
	i = i + 1;
end;

loop 
	int i = 1
until
	i > 7
begin
	upCircBlocks[1].add(scaleA[i]);
	upCircBlocks[2].add(scaleC[i]);
	upCircBlocks[3].add(scaleEb[i]);
	downCircBlocks[1].add(scaleA[abs(i-8)]);
	downCircBlocks[2].add(scaleC[abs(i-8)]);
	downCircBlocks[3].add(scaleEb[abs(i-8)]);
	i = i + 1;
end;

loop
	int r = 1
until 
	r > 15
begin
	randA.shuffle();
	randC.shuffle();
	randEb.shuffle();
	loop
		int t = 1
	until 
		t > 7
	begin
		randomBlocks[1].add(randA[t]);
		randomBlocks[2].add(randC[t]);
		randomBlocks[3].add(randEb[t]);
		t = t + 1;
	end;
	r = r + 1;
end;


loop
	int i = 1
until
	i > blockOrder.count()
begin
	if
		blockOrder[i] == "upScaleA"
	then
		blocks.add(upScaleBlocks[1]);
	elseif
		blockOrder[i] == "upScaleC"
	then
		blocks.add(upScaleBlocks[2]);
	elseif
		blockOrder[i] == "upScaleEb"
	then
		blocks.add(upScaleBlocks[3]);
	elseif
		blockOrder[i] == "downScaleA"
	then
		blocks.add(downScaleBlocks[1]);
	elseif
		blockOrder[i] == "downScaleC"
	then
		blocks.add(downScaleBlocks[2]);
	elseif
		blockOrder[i] == "downScaleEb"
	then
		blocks.add(downScaleBlocks[3]);
	elseif
		blockOrder[i] == "upCircA"
	then
		blocks.add(upCircBlocks[1]);
	elseif
		blockOrder[i] == "upCircC"
	then
		blocks.add(upCircBlocks[2]);
	elseif
		blockOrder[i] == "upCircEb"
	then
		blocks.add(upCircBlocks[3]);
	elseif
		blockOrder[i] == "downCircA"
	then
		blocks.add(downCircBlocks[1]);
	elseif
		blockOrder[i] == "downCircC"
	then
		blocks.add(downCircBlocks[2]);
	elseif
		blockOrder[i] == "downCircEb"
	then
		blocks.add(downCircBlocks[3]);
	elseif
		blockOrder[i] == "randomA"
	then
		blocks.add(randomBlocks[1]);
	elseif
		blockOrder[i] == "randomC"
	then
		blocks.add(randomBlocks[2]);
	elseif
		blockOrder[i] == "randomEb"
	then
		blocks.add(randomBlocks[3]);
	end;
	i = i + 1;
end;


sub
	collectResponse(int b, string bt)
begin
	outData[trial_ct][1] = string(trial_ct);
	outData[trial_ct][2] = string(b);
	outData[trial_ct][3] = blockType;
	outData[trial_ct][4] = trialInfo[1];
	outData[trial_ct][5] = trialInfo[2];
	outData[trial_ct][6] = trialInfo[3];
	outData[trial_ct][7] = trialInfo[4];
	if
		bt.substring(1,2) == "up"
	then
		outData[trial_ct][8] = "up";
		if bt.substring(3,4) == "Circ"
		then outData[trial_ct][9] = "circular";
		else outData[trial_ct][9] = "scale";
		end;
	elseif
		bt.substring(1,4) == "down"
	then
		outData[trial_ct][8] = "down";
		if bt.substring(5,4) == "Circ"
		then outData[trial_ct][9] = "circular";
		else outData[trial_ct][9] = "scale";
		end;
	else
		outData[trial_ct][8] = "random";
		outData[trial_ct][9] = "random";
	end;
end;



loop
   int b = 1
until
    b > blockOrder.count()
	#b > 1
begin
	blockType = blockOrder[b];
	term.print(blockType);
	if blockType.substring(1,6) == "random"
	then
		loop
			int t = 1
		until
			t > blocks[b].count()
		begin
			trialInfo = blocks[b][t];
			wav.set_filename(trialInfo[1]);
			term.print(trialInfo[1]);
			wav.load();
			stim_event.set_port_code(1);
			collectResponse(b, blockType);
			tone_pres.present();
			t = t + 1;
			trial_ct = trial_ct + 1;
		end;
	else
		loop
			int r = 1
		until
			#15 repetitions
			r > 15
		begin
		#repeat sequence of tones for blocktype
			loop
				int t = 1
			until
				t > blocks[b].count()
			begin
				trialInfo = blocks[b][t];
				wav.set_filename(trialInfo[1]);
				wav.load();
				stim_event.set_port_code(1);
				collectResponse(b, blockType);
				tone_pres.present();
				t = t + 1;
				trial_ct = trial_ct + 1;
			end;
			if outData[trial_ct-1][9] == "scale"
			then 
			silence.set_duration(random(500,750));
			silence.present();
			end;
			r = r + 1;
		end;
		if outData[trial_ct-1][9] == "circular"
		then 
		silence.set_duration(random(500,750));
		silence.present();
		end;
	end;
	b = b + 1;
end;

string logPath = "Z:\\MORPHLAB\\Projects\\shepard\\";
string colheaders = "trial\tblock\ttrial_type\twav_file\tcondition\tkey\tfreq\tupdown\tcircscale";
string filename = logPath+"logs\\"+logfile.subject()+"_shepard_"+date_time("yyyymmddhhnn")+".txt";
writeArray(outData, filename, colheaders);