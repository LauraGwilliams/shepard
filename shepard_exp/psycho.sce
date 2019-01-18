scenario = "pitchmatch";
no_logfile = false;
default_font_size = 18;
response_matching = simple_matching;
response_logging = log_active;
active_buttons = 4; 
button_codes = 0,0,0,0;
default_background_color = 105,105,105;
default_text_color = 255, 255, 255;
default_font = "Courier New";
write_codes = true;
pulse_width = 5;

begin;

trial {
	trial_type = fixed;
	trial_duration = 600;
	picture {
		text { caption = "NO"; font="Arial"; } text1;
		x = -150; y = 0;
		text { caption = "YES"; font="Arial"; } text2 ;
		x = 150; y = 0;
	};
	time = 0;
	code = "fix";
} fixation;

trial {
	trial_type = specific_response;
	trial_duration = forever;
	terminator_button = 1,2,3,4;
	picture {
		text { caption = "Block"; } instText;
		x = 0; y = 150;
	};
	time = 0;
	code = "block";
} instructions;

trial { 
	trial_type = fixed;
	trial_duration = 300;
	
	stimulus_event {
		picture {
			text { caption = "+"; font= "Arial"; };
			x = 0; y = 0;
			text { caption = "NO"; font="Arial"; } text3;
			x = -150; y = 0;
			text { caption = "YES"; font="Arial"; } text4;
			x = 150; y = 0;
		} visual;
		time = 0;
		code = "img";
	} fixationEvent;

	stimulus_event{
		sound {
		wavefile {filename = ""; preload = false;} wav;
		} sound;
		time = 0;
		code = "sound1";
	} soundEvent;
} soundPresentation;

trial {
	trial_type = specific_response;
	trial_duration = forever;
	terminator_button = 3,4;
	
	stimulus_event {
		picture {
			text { caption = "NO"; font="Arial"; }text5;
			x = -150; y = 0;
			text { caption = "YES"; font="Arial"; }text6;
			x = 150; y = 0;
		} ques;
		time = 0;
		target_button = 3,4;
		response_active = true;
		code = "imgPres1";
	} questionEvent;
} question;

###################------PCL------######################
begin_pcl;

include "Generic_subroutines.pcl";


int nFields = 3;
array<string> itemSet1[25][3];
array<string> allBlocks[1][0][0];
array<string> blocks[0][150][nFields];
array<string> outData[150][8];
int trial_ct = 1;

array<string> trialInfo[3];
string blockType;
array<string> blockOrder[] = {"aud"};

input_file infile1 = new input_file;
infile1.open("psycho.txt");


loop
	int row = 1
until
	row > 25
begin
	loop
		int col = 1
	until
		col > nFields
	begin
		itemSet1[row][col] = infile1.get_string();
		col = col + 1;
	end;
	row = row + 1;
end;

itemSet1.shuffle();

loop
	int j = 1
until 
	j > 6
begin
	loop
		int i = 1
	until
		i > 25
	begin
		allBlocks[1].add(itemSet1[i]);
		i = i + 1;
	end;
	itemSet1.shuffle();
	j = j + 1;
end;

# loop through randomized order of block types
# append last block to overall block list
# remove the block from its old list so a different one is taken next time
loop
	int i = 1
until
	i > blockOrder.count()
begin
	blocks.add(allBlocks[allBlocks.count()]);
	allBlocks.resize(allBlocks.count() - 1);
	i = i + 1;
end;

# subroutine to collect response data
sub
	collectResponse(int b)
begin
	stimulus_data lastStim = stimulus_manager.last_stimulus_data();
	outData[trial_ct][1] = string(trial_ct); # trial count
	outData[trial_ct][2] = string(b); # block count
	outData[trial_ct][3] = trialInfo[1];
	outData[trial_ct][4] = trialInfo[2];
	outData[trial_ct][5] = trialInfo[3];
end;

string expInst = "In this task, you will see a fixation cross that \n\nsometimes will be and sometimes will not be accompanied by a tone. \n\n\n\nPlease indicate whether YES, you heard a tone (pointer finger), \n\nor NO, you did not (middle finger).";
string switchInst = "From now on, please use your pointer finger to select NO \n\nand your middle finger to select YES.";
instText.set_caption(expInst, true);
instructions.present();
# Do the experiment
loop
	int b = 1
until
	b > blockOrder.count()
	#b > 2
begin
	loop
		int t = 1
	until
		t > blocks[b].count()/2 - 1
		#t > 3
	begin
		trialInfo = blocks[b][t];
		soundEvent.set_port_code(1);
		wav.set_filename(trialInfo[1]);
		wav.load(); 
		fixation.present();
		soundPresentation.present();
		question.present();
		int responseButton = response_manager.last_response();
		stimulus_data last = stimulus_manager.last_stimulus_data();
		int rowReactionTime = last.reaction_time();
		outData[trial_ct][6] = string(rowReactionTime); 
		outData[trial_ct][7] = string(responseButton - 3);
		if (int(outData[trial_ct][7]) == 0 && int(trialInfo[3]) == 0) || ((int(outData[trial_ct][7]) > 0) && (int(trialInfo[3]) > 0))
		then
			outData[trial_ct][8] = string(1)
		else
			outData[trial_ct][8] = string(0)
		end;
		collectResponse(b);
		t = t + 1;
		trial_ct = trial_ct + 1;
	end;
	instText.set_caption(switchInst, true);
	instructions.present();
	loop
		int t = blocks[b].count()/2
		#int t = 4
	until
		t > blocks[b].count()
		#t > 6
	begin
		trialInfo = blocks[b][t];
		text1.set_caption("YES",true);
		text2.set_caption("NO",true);
		text3.set_caption("YES",true);
		text4.set_caption("NO",true);
		text5.set_caption("YES",true);
		text6.set_caption("NO",true);
		soundEvent.set_port_code(1);
		wav.set_filename(trialInfo[1]);
		wav.load(); 
		fixation.present();
		soundPresentation.present();
		question.present();
		int responseButton = response_manager.last_response();
		stimulus_data last = stimulus_manager.last_stimulus_data();
		int rowReactionTime = last.reaction_time();
		outData[trial_ct][6] = string(rowReactionTime); 
		outData[trial_ct][7] = string(-responseButton + 4);
		if (int(outData[trial_ct][7]) == 0 && int(trialInfo[3]) == 0) || ((int(outData[trial_ct][7]) > 0) && (int(trialInfo[3]) > 0))
		then
			outData[trial_ct][8] = string(1)
		else
			outData[trial_ct][8] = string(0)
		end;
		collectResponse(b);
		t = t + 1;
		trial_ct = trial_ct + 1;
	end;
	b = b + 1;
end;

string logPath = "C:\\Documents and Settings\\megadmin\\Desktop\\Presentation Scripts\\Ellie\\pre-tests\\";

string colheaders = "trial\tblock\twav_file\tfreq\tamp_level\tRT\ttone_present\taccuracy";
string filename = logPath+"logs\\"+logfile.subject()+"_psycho_"+date_time("yyyymmddhhnn")+".txt"; 
writeArray(outData, filename, colheaders);