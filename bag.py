import jieba
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# 语料
corpus = np.array([
    "从前有户人家黑瓦顶的洞还没来得及补好，雨就开始以迅猛之势往下倾倒，丝毫不看时节，可真是“屋漏偏逢连夜雨”啊。更不巧的是，这户人家刚好在炒大锅菜，而那屋顶的洞恰好在大锅的正上方。于是便有了下面这幅场景：这户人家的小女儿在厨房里打了一把伞防止漏下来的雨水掉进锅里，二女儿在生火，妈妈在炒菜，大女儿则负责给妈妈打下手。看到这里，我想大多数人可能很难相信这是一个00后的亲身经历",
    "我出生于江西省赣州市的一个偏远且落后的农村，由于地域贫富差距较大，尽管我是一个00后，但是我和现在大部分00后大学生所见识到的世面、童年经历很不一样。反而，我成长的环境和大多数90后的成长环境更为相像——土胚房、茅房、黑瓦屋顶、黑白电视、弹棉花、稻谷脱粒机、老式拉线灯、井水等等，这些东西只有少数在发展落后的农村成长的00后才可能亲身经历。而像大多数00后小时候都熟知的麦当劳、肯德基、4399小游戏、芭比娃娃、电影院、游乐园等在我的童年是空缺的。虽然现在本科学历贬值严重，但是在我的家乡能考出一个上一本的大学生却仍是一件稀罕事",
    "在全面建成小康社会的大征程中，少数积贫积弱地区的发展速度很难跟上大部队的步伐。“马太效应“同样适用于地区与地区之间，即相比之下贫困区会更贫困，发达地区会更发达。纵向来看，贫困地区近二三十年变化十分缓慢，最为显著的变化也不过于多了几户人家把土胚房拆了建起了红砖房；而从横向看，同样的发展时间内，其他发达及较发达地区的改变已日新月异。在这种背景下，农家子弟通过教育实现阶层跨越的旅程与几十年前农家子弟的求学之路会有哪些不同？是否会变得更加困难？现今的“寒门子弟”会有什么样的情感体验？我本人作为出生于贫困区的大学生之一，我的教育经历就是现今“寒门子弟”们的一个缩影，在一定程度上可以解答以上问题。",
    "2009年秋到2011年夏，我在村里上小学一二年级。这所村小的历史还算悠久，我爸就是在这儿上的小学，教他的老师也教过我。整个学校只有一层，但一连有七八个教室，一排教室前面有一小块水泥坪，那就是我们的操场，上下课的铃声是以老师的敲钟声为准。听爸爸说，他上学那会儿这个学校一共有五个年级，可等到我来这上一年级时，全校只有三个年级，每个年级只有十几个人。全校一共只有语文、数学两个老师，一二年级不用上英语课，只有三年级需要上，他们的英语老师是镇里的小学派过来的，一周只用上一节英语课，那时每次看到三年级的哥哥姐姐们上完英语课后都在用英语开玩笑就觉得很有趣，便开始期待我升三年级后的英语课，可谁知，轮到我上三年级时这所小学把三年级废除了（现在这所村小已经被改建成了村委会），我之后只好转学去镇上的小学。由于村小的学生太少了，老师上课采用的是复式教学的组织形式，即一二年级共用一个教室（剩下的五六个教室都荒废了），这边教完一年级并让他们先做着练习，转而就开始教教室另一边的二年级。全校两个老师上课几乎都是用方言，因为老师们自己也不太会讲普通话。于是问题就来了——用方言教汉语拼音大概率是很难把学生的发音教标准的……这也就是我平翘舌不分、前后鼻音不分、边音卷舌音不分的主要原因。",
    "其实我小学一年级时成绩并不好，而且十分贪玩，一放学就和村里的小朋友出去玩，把老师布置的作业抛在九霄云外。但是有一天妈妈突然检查我的作业，让我把语文课上学的生字拼读给她听，但是我支支吾吾了半天也拼不出来，妈妈见我学习如此不用心，一怒之下用绳子把我整个人捆在椅子上，我根本无法动弹，仍由她拿鞭子抽得我嗷嗷叫，这种皮肉之痛实在不敢再经历第二次。我当初很不理解妈妈为什么要对我这么狠心，现在回看才知道妈妈是怕我不争气，重蹈她的覆辙，一辈子都困在小小的农村干农活累活。自那天起我便开始认真读书，生怕哪一天妈妈问我的问题我答不上来又挨一顿打。功夫不负有心人，一二年级我的学习成绩越来越好，我成为了左邻右舍口中那个“别人家的孩子”。但接下来的转学对我来说打击不小。",
    "2011年秋到2015年夏，我来到镇上的小学上三到六年级。镇小的学生规模是村小的几十倍，有六个年级，每个年级有五六个班，每班有五六十个人，有那么多同学和我一起学习我觉得很新奇，但我并不能很快地适应新的教学模式，上午大课间要做广播体操，下午大课间要做眼保健操，且都会有少先队员来检查做操情况。但是我哪会做广播体操和眼保健操呢？村小的老师根本没有教过这些东西。我怕自己被其他同学嘲笑，也怕拖了班级的后腿，就偷偷观察身边的同学，有样学样，日复一日，我终于在三年级上学期期末之前学会了。但是我转学后的成绩却一落千丈，难有起色。老师布置的作业太多了，我积累了一堆作业没写。",
    "在发展落后的农村，“重男轻女”的封建思想往往根深蒂固，我的家乡也不例外。姑姑有个儿子和我同级同校但不同班，我妈妈生了三个女儿，爷爷奶奶对我们姐妹三个不是很待见，总是更疼爱姑姑的儿子。因为我家离镇小有点远，而姑姑家恰巧在学校对面，所以刚转学那会儿妈妈担心我不熟悉环境，就让我每天中午放学去姑姑家蹭饭，但每次吃饭都少不了一番比较，好几次月考表哥都90多分，我只有70多分，三年级上学期期末我没有像以前一样拿到“三好学生”的奖状，那年过年姑姑狠狠炫耀了一番表哥的奖状，就这样我这个“别人家的孩子”人设崩塌。我心里的不服气越来越浓，心想我一定要争气，为妈妈争光，不能让别人看低我们家。于是我便憋着一股劲努力学习，老师上课问的问题我总是回答得最大声，慢慢地我的成绩提上去了，同时我的进步也被班主任看在眼里。老师开始注意课上回答问题最认真最响亮的那个声音，开始注意考试成绩进步很明显的那个名字，终于在一次班会上，他把我叫到讲台前，在全班同学的面前表扬了我的认真与勤奋，甚至开玩笑说我这么努力下去可以“上清华北大”，虽然最后事实证明我没有上清华北大的天赋，但是老师这一番肯定给予了一个小学三年级的女孩莫大的鼓励，我开始变得自信起来。从某种程度上来说，我小学三年级班主任是我遇到的“重要他人”之一，我每每想到他对我的表扬和期许都能充满希望、继续不忘初心地努力学习。三年级下学期，我终于重获“三好学生”的奖状，自那起我的学习成绩就一直保持在班级前五。由此可见，“农家子弟”学习动力很大一部分来源于为父母争光，尤其是女娃，会想扬眉吐气，堵住那些说“女子不如男”的嘴。",
    "小学三到六年级虽然有英语课，但是一周只有一两节，期末也不会考。学校没有专门的英语老师，是英语比较好的数学老师或者语文老师来上课。如果没有领导来检查的话，英语课就顺理成章地成为了自习课，所以我的小学基本上是没有怎么学英语的，是在上初中后才真正开始学。镇小比村小的教学设施好很多，有呼啦圈、乒乓球桌、毽子等体育器材，有正式的升旗台，但还没有塑胶跑道和班班通。此外，除了上课，大家是用普通话交流，其他时间都是用方言，在这样的环境下，我不标准的普通话发音虽然没有被同学嘲笑，但同时也没有被好好纠正的机会。",
    "2015年秋到2018年夏，我在镇里唯一的中学上初中。爷爷在我小升初的那个夏天去世了，家里只剩下爸妈和我们三姐妹五个人，爸爸妈妈的文化水平都是小学，只能从事一些体力劳动，爸爸那时候有一份在砖厂里搬红砖的工作，一个月3千块钱。我的妈妈由于上世纪七八十年代农村医疗条件极差，小时候去看病，右脚打针打坏了发生严重的萎缩变形，自那起妈妈再也没有办法像正常人一样走路，学业也因此落下了很多，读到四年级就因为没钱读书就辍学了。妈妈虽然没办法从事太重的体力活，但她为了供我们三姐妹上学依旧忍痛通过种菜去挣钱，我偶尔也跟着妈妈去菜市场卖菜，从早守到晚也挣不到一百块钱，而且菜也不是天天有的卖。爷爷在世的时候，他的退休金还可以为家里缓解一下燃眉之急，爷爷去世之后，家里的经济来源就是爸爸一个月三千的工资和每个月的低保等政府补助。所以上初中后，我愈发理解家里拮据的经济情况以及父母的压力山大，甚至在想自己就算成绩好又有什么用，还是没办法帮父母分担，这种无奈感越来越强烈，曾一度让我萌生出辍学去赚钱的想法。但母亲一直叮嘱我家里的事情我不用担心，她会处理好，我只需要好好读书考出这个小农村去见见世面。我的母亲不是无所不能，但为了我们，她真的做到了无所畏惧。小时候家里还是住的土胚房，我、妹妹和妈妈挤在一张床上，姐姐住校，爸爸睡砖厂，爷爷在另一个屋子里睡。有天半夜妈妈起来开灯上厕所发现门口有一双陌生男子的脚迟迟不动，很明显，门外站的是……强盗。我和妹妹害怕得不敢吱声，要知道土胚房的门是个木门，只要门外的男人力气足够大，他完全可以撞开。那时家里只有爷爷一个男人，且爷爷年事已高，根本无法与之抗衡，但我的妈妈就像一个超级英雄一样十分冷静，她说不要怕，有她在。经过快速的分析，最好的解决方法就是把动静闹大，让邻居都听到，然后把门外的强盗吓走。于是这个瘦小的女人，竭尽全身力气吼出了所有男性邻居的名字，试图让他们听到然后过来帮忙，小小的身躯竟然可以爆发出如此铿锵有力的声音，响亮到可以跨越时空，让十几年后的我还能听到。那个声音没有丝毫畏惧，没有丝毫颤抖，有的只是一位母亲保护孩子的坚定。在我所有童年的记忆里，无论春夏秋冬，无论风多么凛冽，水多么刺骨，只要我在家，早上都一定能吃到热腾腾的早饭。我的母亲迈着一轻一重的步伐，一次又一次地用实际行动打消了我想辍学打工的念头。我开始比以前更加拼命地学习。",
    "由于中考要考体育，初三那年学校抓体育锻炼抓得特别紧，所以我从初三开始住校。每天早上5点50集合开始跑步，初中没有塑胶跑道（所以之后我们要从镇里来到区里有塑胶跑道的中学考体育），我们都是绕着教学楼跑，即使是冬天也要5点50准时集合，跑步都是摸黑跑，得跑好一阵天才会亮。每天我都严格要求自己训练每一个项目，但是由于学校组织的训练强度太大了，我的膝盖出现了一些问题，经常跑着跑着就一阵剧痛，严重时我几乎走不了路，但是我那时怕去医院检查太花钱，就没敢和家里人说。一般来说，只要我跑800米的时候腿疾没有发作，我就几乎是组内第一个到达终点的。天道酬勤，全校只有两个女生中考体育满分，我是其中之一，最后的中考也超常发挥，总成绩全校排名第二。中考后的那个暑假，妈妈带我去检查膝盖，医生说是半月板变形，里面有积水，变形是不可逆的，无法根治，只能日后多加小心。我试过针灸，贴膏药，熏艾草等各种法子，确实只能缓解，无法根治。每当我走路走久了开始腿痛膝盖痛时，都会问自己：用膝盖的健康来换一个体育满分，真的值得吗？但要是重来一次，我还是会那么做。对于当时的我来说，没有什么值不值得，因为摆在我面前的只有一条路，就是拼尽全力考上区里的重点高中。",
    "2018年秋到2021年夏，我如愿来到区重点高中读书。区里只有三所普高，其他都是职高，我努力进入的是区里最好的高中里的重点班。由于膝盖受伤，我没有办法参加军训，也因此失去了熟悉新同学的最佳机会。高中不像小学初中，大家除了上课，平时交流都是用普通话，这是我来到新环境的第一个冲击，我很难自然地切换成普通话模式，经常和同学讲着讲着话就会不自觉地蹦出几句方言来。班上的同学都多才多艺，各展风采，挥斥方遒，知识储备也很丰富，对他们来说可能是常识的东西我可能是第一次听说。和我的同学们相比，我感觉自己就像个书呆子，没有什么兴趣爱好，也没什么特长，这是我上高中之后遇到的第二个冲击。江西是在2021年才开始实行新高考“3+2+1”政策，而我上高一时是2018年，所以和以往一样是文理分科。高一上学期要学9门科目，每门科目都有很多作业，我根本做不完，上课听不懂，课后自学也学不明白，我们高中每个月都有一次年级大考，每个班平均50多个人，而我每次的班级排名都是四五十名，我不再像初中时那样成绩名列前茅，最可怕的是无论我怎么努力，我的成绩都上不去，这种巨大的落差与无力感打我一个措手不及，加上我住校，家人长期不在身边让我更加孤独，所有这些情绪聚合成一只猛禽无情地扑咬我，我就这样自卑又焦虑地度过了我的高一上学期。",
    "高一上学期期末文理分科，我选择了理科。原因很简单，背历史政治的知识点对我来说是莫大的折磨，背完这一个知识点，上一个知识点就忘得差不多了，上历史课就像是在听天书，老师提到的那些术语历史典故什么的我都不了解。选了理科后，只用学6个学科，任务轻松了不少，我开始能够做完老师布置的作业，并且能够及时反思自己的学习方法。我实在不喜欢处于低谷的感觉，迫切地想要提高我的成绩，于是新学期我强迫自己变得更加积极，逼着内向的自己去请教同学和老师们问题，每次上课都会提前预习新知识，以防自己上课又听不懂，除了学校发的练习题，我还额外买了很多题来刷，功夫不负有心人，我终于不再是倒数几名，我的排名一直在上升。",
    "高二是我成绩提升的关键期，全部老师都换了，而我又遇到了一个“重要他人”——我的物理老师。高一升高二的那个暑假，由于我的物理太差了，我开始着重自学物理，买了一本辅导书，每天都学一点，并完成学校提前发下来的物理练习册，我很享受自学的过程，因为我完全可以按照自己的节奏去学习，慢慢把每一个知识点都琢磨清楚，非常有成就感。我开始期待新学期的到来，期待我之后的进步。高二的物理是一位带过好几批实验班的资深老师教的，开学第一个月他就收集了全班的练习册来检查，想大概了解每个同学的学习情况，由于我的练习册超前完成了很多，自然他对我的印象就十分深刻，在下节课当众表扬了我。向来物理差的我，受到表扬后像是打了鸡血一样学物理，我没有想到上高中以来一直不起眼的我也能被老师注意到。物理老师应该是看出来了我是一个很内向、不太敢主动问问题的学生，课后还单独找我谈过几次话，会关心我平时学习会有什么疑问，有任何不懂的题目都可以来问他，我真的受宠若惊。一个老师对学生的影响是巨大的，一个老师不经意的几句话，甚至几句简单的寒暄，就足以给一个自卑又内向的学生带来巨大的改变。所以贫困区的教师质量的提升，对小到青少年的成绩进步和大到全面建成小康社会都是至关重要的。",
    "笨鸟先飞，高二一年每次大考我都会认真分析并总结错题，虚心向成绩比我好的同学请教，所以我的成绩进步非常大，基本保持在班级前10名，我逐渐把目标锁定在班级前三，高二上学期的期末考试我现在还记忆犹新，因为那是我上高中以来第一次全班第一，这种成就感和喜悦感是溢于言表的，我终于熬过那一年多的低谷，见到了黎明的曙光。高二寒假过年那会儿疫情爆发，学校延迟开学，采用的是线上授课的方式，那时全家除了在上大学的姐姐有手机和电脑外，全家就只有妈妈一个人有手机（我的爸爸是智力残疾人，所以他不用手机），但妈妈的手机用了很多，内存很小，上网课非常卡顿，坚持了一两周之后我干脆就不听网课了。在极其自由的线上授课阶段，每个科目的作业都很少，我每天都有很多时间去自学，所以在高二格外漫长的寒假我学完了高二下的有机化学，开学后化学老师讲的知识点，其他人可能第一次接触，但我已经背得滚瓜烂熟。人不可能永远处在低谷，但也不能一直处在巅峰，经过了名列前茅的高二后，我来到了我的瓶颈期——高三复习。我像一只泄了气的气球，没有往上飞的劲儿了，不知道给飞向哪里，很迷茫。而此时，我身边的同学收起了高一高二的玩心，开始认真复习，进步特别大，仿佛他们只用好好学习一个学期就能考到我努力了两年多才取得的成绩，很快我就被赶超了。到了高三下学期，学校采用两天一考的模式来训练学生的做题速度，一天考试一天讲题，但是一天根本不够我仔细研究错题，前一天的题目还没弄懂，紧接着后一天又要迎接新的题目，不会做的还是不会，这种题海战术让我感到窒息与倦怠，现在回想起来还是觉得这个模式非人性。但这种模式有一种好处就是让我对考试已经完全麻木了，高考对我而言就像每周的小考一样平平无奇，我保持着一种平常心去迎接高考。",
    "高考最后一门考完，我很清楚会和以往的考试一样考得不是很好，所以我一直没敢对答案，也没有估分，反正无论结果如何我都不会选择复读，因为高三那种高强度的复习模式已让我精疲力竭，我不想再来一次。煎熬地等待了一个多月后，高考成绩出来了，确实考得一般。由于我本人没有什么很特别的兴趣爱好，所以填报志愿的时候没有很心仪的专业。虽然在疫情期间对医生这个职业充满了崇拜，也有考虑过学医，但是学医要读很多年才能出来赚钱，家里几乎不可能维持我这么多年的学费，所以很快也就打消了这个念头。大我五岁的姐姐刚好也上的华中师大，和我一样学的理科，她一开始的专业是电子信息类，但由于高数和大学物理太难学她之后转去了日语专业，所以在填报志愿的时候她强烈建议我不要学理工科的专业，怕我没有那种天赋会挂科。在姐姐的建议下，我决定选不用学高数的专业，便把华师的英语公费师范作为第一专业志愿，翻译作为第二专业志愿，但最后分不够，录的是翻译专业。",
    "2021年秋我来到华师上大学，这是我第一次出省，来到新的城市我既兴奋又忐忑，这里有很多我没见过的东西，我第一次坐火车，第一次坐地铁，第一次坐电梯，第一次喝奶茶……各种体验都很新奇，但体验的过程中我很害怕，害怕自己没有这方面的常识而被人嘲笑。第一次坐地铁时我给自己加油鼓劲了好一会儿才敢跨出第一步，从这一件小事足以看出我大学生活做的每一次尝试前都必经历一次又一次的心理斗争，所以我经常陷入精神内耗，会怀疑自己的能力，质疑自己的价值，这可能就是现在的农家子弟与以往的农家子弟最大的不同——有着很严重的畏难情绪与精神内耗。但由于我内心深处不服输的劲和想要为父母争光的志向还没有消亡，所以无论我多么害怕与焦虑，我最终还是会逼着自己去参加比赛，去勇敢尝试，因为不去尝试就什么也没有了。",
    "大学提供了一个更高的平台，让我有机会可以和那么多优秀的同学一起学习，在大一开学的头几次年级大会上同学们在竞选年级学生会干部，每个人都神采奕奕地上台讲述自己丰富多彩的经历和技能，他们身上散发的那种自信的光芒令我向往，他们有的钢琴十级，有的获得过各种奥赛奖，有的会弹古筝，相比之下，在台下的我除了学习什么也不会，我又开始不自觉地感到自卑。但另一边我又在给自己做心理建设，帮助自己逃出自卑的掌控。家里能供我上大学已经算是奇迹，根本没有条件让我去上兴趣班，爸妈已经在他们力所能及的范围内给了我最好的教育资源，我不应该抱怨环境，而是努力运用身边现有的一切资源提升自己，改变命运，“各有姻缘，莫羡人”。",
    "翻译专业是自付学费，虽然学费可以用贷款交，但是武汉的物价比家里高很多，平时的生活费对我家来说仍旧是很大的负担，大一那会儿我省吃俭用，每天花费控制在20元左右，但这么下去不是个办法。一方面，由于从小到大我遇到了很多良师，教师对我来说是一份可以帮助更多人改变命运的职业，我对它充满了敬畏与热爱；另一方面，翻译专业的就业前景越来越不理想，我便下定决心要转专业去英语公费师范，就这样我大一上学期学有余力时都在准备转专业笔试和面试。英语转专业考试竞争非常激烈，对我而言无异于第二次“高考”，好在最后我转专业成功了，这意味着爸妈肩上的担子可以轻一些了，并从姐姐那里得知了华师的逸华基金会，我成功申请到了逸华助学金，同时也遇到了我人生中的又一个“贵人”——逸华基金会的发起人高新民教授。高教授看了我的感谢信之后，了解到我家极其贫困的情况，决定再资助我一笔助学金，这让在黑暗中反复挣扎的我感受到了无比的温暖。",
    "转专业之后我还是很认真地学习，英语发音不好我就每天早起到露天电影场练习发音，逼着自己去参加英语角，和外国人交流，我的中式发音终于得到了改善。但我逐渐意识到了自己的问题，我的学习太过功利，我是为了学分绩而学，并不是因为对知识的渴望才学，这让我很容易被学分绩裹挟着走，老师给分稍微低一点都能让我焦虑很久，虽然我大二一年学分绩排名前百分之十，但我讨厌被学分绩捆绑着学习，我享受自由、主动地学习，而不只是为了考试拿高分。所以上大三之后我把我的目光从学分绩转移了，我不想把大学过成是高三的升级版，我不想当一个学习的机器，我想平衡好生活与学习，我没有钱像其他同学一样来一场特种兵旅行，但我可以去感受武汉本地的风土人情，去江滩吹风，去东湖绿道骑车，去小吃街享受美食，走出社恐的舒适圈，走到人群中去，感受人与人之间的关怀。大三，我终于开始学会松弛，不再把自己的缺点和别人的优点作比较，不再紧绷着自己专注在学习上，不再因为一件小事就开始慌张，而是学会放下，锻炼出可以从容接受一切结果的心态。",
    "回首我的教育故事，这一路走来我要感谢的人太多，感谢我的父母供我读书，感谢我遇到的良师益友给予我的帮助，感谢自己一次又一次地救自己与水火之中，没有因为原生家庭而自暴自弃、怨天尤人，或许将来我会遇到更多风雨坎坷，但我仍旧会把那份笨拙的努力坚持到底。",
    "对于我的职业生涯规划，我毕业后大概率是回江西教书，这场阶层旅游或许又会回到最初的起点，我会继续生活在一个不富裕的地区，但不可否认的是，我确实通过读书这条路成功走出去了，见识到了外面世界的多姿多彩，拓宽了视野，未来我会尽我所能通过教育来帮助更多像我这样的农家子弟走出贫困的牢笼，为滚滚向前的中国教育贡献自己的一份绵薄之力。"
])

ll = []
for text in corpus:
    words = [word for word in jieba.cut(text)]
    ll.append(' '.join(words))

# 将文本中的词转换成词频矩阵
vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(ll)

print('词汇表:\n', vectorizer.vocabulary_)

print('词向量矩阵:\n', bag.toarray())

tfidf = TfidfTransformer()

tfidf = tfidf.fit_transform(bag)

print('tfidf向量矩阵：\n', tfidf.toarray())

lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')

lda.fit_transform(tfidf)

print(lda.components_.shape)

n_top_words = 5
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
