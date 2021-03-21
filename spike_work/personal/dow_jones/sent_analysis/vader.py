from nltk.sentiment import SentimentIntensityAnalyzer

headline_list = """Most cases of cancer are the result of sheer bad luck rather than unhealthy lifestyles,
 diet or even inherited genes, new research suggests. Random mutations that occur in DNA when cells 
 divide are responsible for two thirds of adult cancers across a wide range of tissues. Iran dismissed 
 United States efforts to fight Islamic State as a ploy to advance U.S. policies in the region: "The 
 reality is that the United States is not acting to eliminate Daesh. They are not even interested in 
 weakening Daesh, they are only interested in managing it" Poll: One in 8 Germans would join 
 anti-Muslim marches UK royal family's Prince Andrew named in US lawsuit over underage sex 
 allegations Some 40 asylum-seekers refused to leave the bus when they arrived at their 
 destination in rural northern Sweden, demanding that they be taken back to Malm or 
 "some big city". Pakistani boat blows self up after India navy chase. All four 
 people on board the vessel from near the Pakistani port city of Karachi are 
 believed to have been killed in the dramatic episode in the Arabian Sea on 
 New Year's Eve, according to India's defence ministry. Sweden hit by third mosque
  arson attack in a week 940 cars set alight during French New Year Salaries for
   top CEOs rose twice as fast as average Canadian since recession: study Norway 
   violated equal-pay law, judge says: Judge finds consulate employee was unjustly 
   paid $30,000 less than her male counterpart Imam wants radical recruiters of Muslim
    youth in Canada identified and dealt with Saudi Arabia beheaded 83 people in 2014,
    the most in years 'A living hell' for slaves on remote South Korean islands - Slavery
      thrives on this chain of rural islands off South Korea's rugged southwest coast, 
    nurtured by a long history of exploitation and the demands of trying to squeeze a 
    living from the sea. Worlds 400 richest get richer, adding $92bn in 2014 Rental Car 
    Stereos Infringe Copyright, Music Rights Group Says Ukrainian minister threatens TV 
    channel with closure for airing Russian entertainers Palestinian President Mahmoud 
    Abbas has entered into his most serious confrontation yet with Israel by signing onto 
    the International Criminal Court. His decision on Wednesday gives the court jurisdiction 
    over crimes committed in Palestinian lands. Israeli security center publishes names of 50 
    killed terrorists 'concealed by Hamas' The year 2014 was the deadliest year yet in Syria's 
    four-year conflict, with over 76,000 killed A Secret underground complex built by the Nazis 
    that may have been used for the development of WMDs, including a nuclear bomb, has been 
    uncovered in Austria. Restrictions on Web Freedom a Major Global Issue in 2015 Austrian 
    journalist Erich Mchel delivered a presentation in Hamburg at the annual meeting of the 
    Chaos Computer Club on Monday December 29, detailing the various locations where the US 
    NSA has been actively collecting and processing electronic intelligence in Vienna. Thousands 
    of Ukraine nationalists march in Kiev Chinas New Years Resolution: No More Harvesting Executed 
    Prisoners Organs Authorities Pull Plug on Russia's Last Politically Independent TV Station"""

sing_headline1 = 'Russia Today: Columns of troops roll into South Ossetia; footage from fighting (YouTube)'
sing_headline2 = "Pakistan's Musharraf to Resign, Leave the Country"
sing_headline3 = 'U.S. Navy Ships Head to Georgia'

neg_headline1 = 'EU Stocks plunge!'
neg_headline2 = "Iran leader says 'American empire' near collapse"
neg_headline3 = 'Military coup in Niger'
neg_headline4 = 'Stocks collapse on global slowdown fears'
pos_headline1 = 'CIA shuts down its secret prisons'
pos_headline2 = 'France will not ban Wi-Fi or Tor, prime minister says: "Internet is a freedom, is an extraordinary means of communication between people, it is a benefit to the economy," Valls added'
pos_headline3 = 'Irish economy grows at fastest pace since 2000 at 7.8%'
pos_headline4 = 'Arctic ice begins regrowth'
pos_headline5 = "Hamas accepts Israel's right to exist"

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(headline_list))
print(sia.polarity_scores(sing_headline1))
print(sia.polarity_scores(sing_headline2))
print(sia.polarity_scores(sing_headline3))

print('\n Neg Headlines')
print(sia.polarity_scores(neg_headline1))
print(sia.polarity_scores(neg_headline2))
print(sia.polarity_scores(neg_headline3))
print(sia.polarity_scores(neg_headline4))

print('\n Pos Headlines')
print(sia.polarity_scores(pos_headline1))
print(sia.polarity_scores(pos_headline2))
print(sia.polarity_scores(pos_headline3))
print(sia.polarity_scores(pos_headline4))
print(sia.polarity_scores(pos_headline5))
