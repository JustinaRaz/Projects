<!-- image -->

Contents lists available at ScienceDirect

## Neurocomputing

[journal homepage: www.elsevier.com/locate/neucom](https://www.elsevier.com/locate/neucom)

## A review of IoT applications in healthcare

Chunyan Li a , Jiaji Wang b , Shuihua Wang b,c , Yudong Zhang b,d,*

- a School of Informatics, Yunnan Normal University, Kunming 650500, China
- b School of Computing and Mathematical Sciences, University of Leicester, LE1 7RH, United Kingdom
- c Department of Biological Sciences, Xi ' an Jiaotong-Liverpool University, Suzhou, Jiangsu 215123, China
- d Department of Information Systems, Faculty of Computing and Information Technology, King Abdulaziz University, Jeddah 21589, Saudi Arabia

A R T I C L E  I N F O

Keywords: IoT Healthcare

Remote monitoring

## 1. Introduction

The Internet of Things, abbreviated as the IoT, is a revolutionary and rapidly evolving technology that has the promise to change the way we interact with our surroundings [1 - 3]. It is estimated that by 2025, there will be over 75 billion connected devices worldwide [4]. The healthcare industry is one of the sectors that is expected to benefit greatly from this growth.

IoT involves the interconnection of different kinds of physical sub­ jects, types of equipment, and sensors via the Internet, enabling them to communicate with each other and share data [5,6]. In healthcare, IoT technologies are already being used to improve patient care and out­ comes, offering new  opportunities for remote  monitoring [7,8], personalized treatment plans [9], and efficient healthcare delivery [10]. The  IoT  is  also  transforming  industries  beyond  healthcare,  such  as manufacturing [11], transportation [12,13], and agriculture [14], and has the potential to reshape our society as a whole. As such, it is crucial to  stay  abreast  of  the  latest  developments  in  this  exciting  field  and explore the ways in which IoT can be leveraged for positive change.

In addition to improving patient care, the IoT can substantially and greatly reduce healthcare prices by streamlining processes, automating routine tasks, and decreasing the need for expensive interventions. It enhances patient care by enabling real-time data collection and moni­ toring,  thereby  improving  response  times  and  patient  outcomes.

A B S T R A C T

Integrating Internet of Things (IoT) technologies in the healthcare industry represents a transformative shift with tangible benefits. This paper provides a detailed examination of IoT adoption in healthcare, focusing on specific sensor types and communication methods. It underscores successful real-world applications, including remote patient monitoring, individualized treatment strategies, and streamlined healthcare delivery. Furthermore, it delves into the intricate challenges to realizing the full potential of IoT in healthcare. This includes addressing data security concerns, ensuring seamless interoperability, and optimizing the use of IoT-generated data. The paper seeks to inspire practitioners and researchers by highlighting the practical implications of IoT in health­ care, emphasizing the ways IoT can enhance patient care, resource allocation, and overall healthcare efficiency.

Secondly, IoT technologies streamline healthcare operations by auto­ mating administrative tasks, which not only saves time but also reduces the need for an extensive administrative workforce, resulting in cost savings. Moreover, routine tasks like vital sign monitoring and medi­ cation dispensing can be automated, decreasing the risk of errors and lowering  labor  costs.  Most  significantly,  IoT ' s  early  detection  and continuous  monitoring  capabilities  help  in  averting  costly  medical complications, reducing hospitalizations, and mitigating the need for expensive  interventions,  ultimately  lowering  healthcare  costs  signifi­ cantly. It can also improve patient engagement [15] by giving patients greater control over their healthcare, allowing them to track their health data  and  communicate  more  effectively  with  healthcare  providers. Wearables and apps monitor vital signs, while patient portals offer ac­ cess to medical records. Telemedicine and remote monitoring facilitate virtual appointments and real-time data sharing. Mobile apps help in self-management and data sharing, and secure messaging systems pro­ mote  direct  communication  with  healthcare  providers.  IoT  devices automatically  transmit  data.  Education  and  training  are  essential  to ensure users can confidently navigate these technologies. By utilizing these tools, individuals become active participants in their healthcare, fostering better communication, engagement, and, ultimately, improved health outcomes.

The use of IoT in healthcare brings with it a host of challenges and limitations  that  must  be  carefully  navigated.  These  include  concerns C. Li et al.

* Corresponding author at: School of Computing and Mathematical Sciences, University of Leicester, LE1 7RH, United Kingdom. E-mail address: yudongzhang@ieee.org (Y. Zhang).

<!-- image -->

<!-- image -->

about data privacy and security [16], interoperability issues [17], and the need for standardized protocols and regulations [18]. Data privacy and security concerns are paramount, as the collection and transmission of sensitive medical data require stringent safeguards to protect patient privacy. Interoperability issues can arise due to the multitude of vendors and technologies involved in the development of IoT devices, which can lead  to  data  silos  and  inefficient  communication  [19].  Standardized protocols and regulations are also critical to ensure that IoT devices can seamlessly integrate with existing healthcare systems and networks.

Overall,  the  IoT  is  an  exciting  technology  that  is  important  for healthcare providers, policymakers, and researchers to stay abreast of the latest developments and work collaboratively to address the chal­ lenges  and  limitations  associated  with  its  use.  This  paper  aims  to contribute to this effort by providing a comprehensive overview of the IoT in healthcare and its potential to transform the healthcare industry.

## 2. IoT technologies in healthcare

## 2.1. Sensors

Sensors are an essential component of IoT devices. They can measure and  transmit  data  from  various  sources,  such  as  temperature,  blood pressure, heart rate, glucose levels, and more. These sensors can be in­ tegrated into wearable devices, medical equipment, and even inside the human body. Tables 1 - 4.

Apart from these well-established sensor technologies, researchers have been striving to create more cutting-edge ones.

The VITALS system [20] is an online life-signs sensor that uses four physiological indicators perceived by clinical equipment. It includes a microcontroller, a wireless communication module, and a system for big data analysis. Life signs are extracted and transmitted to the analysis system every half an hour. The system was implemented for one week during  the  experiment  and  demonstrated  improved  care  options, particularly for those with restricted access to medical care. The pro­ posed heart rate sensor [21] utilizes the MAX30102 sensor, which emits visual light in the 650 - 670 nm wavelength range. The sensor is worn on the wrist and records the heart rate signal in the shape of a compara­ tively flat wave, which is then processed using the detrend method to obtain precise data. This research [22] aims to develop a novel sensor using  micro  pyramid-assisted  piezoelectric  film  (MPF)  technology  to constantly monitor blood pressure non-invasively. The results show that the results measured in practice with this MPF sensor are comparable to those  obtained  with  the  more  proven  industrial  products  available currently. This study [23] presents a new method for continuous blood pressure  monitoring  that  requires  the  use  of  only  one  signal  and  no additional sensors and does not require calibration. A pulse oximeter has been created to measure heart rate, blood oxygen saturation, and body temperature [24]. The device uses infrared light and sensors to detect body temperature without contact and displays it on an LCD screen and web browser. The paper [25] describes a novel paper-based capacitive humidity sensor that achieves ultra-high humidity sensitivity and high

Table 1

Common sensors.

Table 2 A comparison of research on remote monitoring.

| Sensor Type       | Measurement                                       | Clinical Application (s)                            |
|-------------------|---------------------------------------------------|-----------------------------------------------------|
| Electrocardiogram | Electrical activity of the heart                  | Detecting arrhythmias, heart disease                |
| Pulse Oximeter    | Oxygen saturation levels and heart rate           | Monitoring oxygenation during surgery, COPD, asthma |
| Blood Pressure    | Blood pressure                                    | Monitoring hypertension and hypotension             |
| Glucose           | Glucose levels in the blood or interstitial fluid | Monitoring diabetes                                 |
| Temperature       | Body temperature                                  | Detecting fevers or hypothermia                     |
| Respiratory       | Respiratory rate and rhythm                       | Monitoring breathing disorders such as sleep apnea  |

Table 3

| Reference   |   Year | Communication   | Advantage (s)                                                                                                                                                             |
|-------------|--------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [95]        |   2022 | Wifi            | ● Continuous monitoring of vital health indicators ● Prompt resolution of issues by healthcare professionals.                                                             |
| [96]        |   2022 | Cloud           | ● Alert Notification for Abnormal Events and Emergencies ● Validation of Effectiveness and                                                                                |
| [97]        |   2022 | Wifi            | Reliability ● Integration of IoT and medical promotion platform ● Evaluation of accuracy and performance ● Identification of optimal body parts for                       |
| [98]        |   2022 | Cloud           | temperature monitoring ● Promotion of the integration of IoT and healthcare                                                                                               |
| [99]        |   2023 | -               | ● Remote delivery of healthcare services with expert support, ● Efficient data collection and mining ● The overall enhancement of patient health surveillance efficiency. |
| [100]       |   2022 | Cloud           | ● IoT Cloud Server Integration ● Real-Time Data Collection and Analysis ● Alert Mechanism                                                                                 |
| [101]       |   2023 | Cloud           | ● Data Input Variety ● Focus on Post-Covid scenarios ● Co-morbidity monitoring                                                                                            |
| [102]       |   2023 | Cloud           | ● Focused on emergency medical services ● Dependability and high success rate for IoT data broadcasts ● Exciting performance with high                                    |
| [103]       |   2023 | Cloud           | accuracy in vital sign ● Store and learn knowledge for future state prediction ● Achieves an impressive accuracy rate of 99.53 %                                          |
| [104]       |   2023 | Wifi            | ● Influenza Symptom-Based Detection ● Broad Application Scope                                                                                                             |

A comparison of research on locating.

| Reference   |   Year | Target               | Advantage (s)                                                                                                            |
|-------------|--------|----------------------|--------------------------------------------------------------------------------------------------------------------------|
| [105]       |   2023 | Patients             | ● Development of a UAV-based radar array system ● Weighted CRB-based source localization                                 |
| [106]       |   2023 | Healthcare Resources | ● Replacement of the existing barcode system ● Smart specimen transport box                                              |
| [107]       |   2023 | Patients             | ● Cloud-based monitoring ● Integration of multiple sensing technologies ● User-friendly interface                        |
| [108]       |   2023 | Patients             | ● Supporting complex functionalities and address environment-wellness relationships ● Environmental sensing and wellness |
| [8]         |   2023 | Patients             | monitoring ● Tracks critical health parameters ● Provides timely medical decisions based on abnormal values              |
| [109]       |   2023 | Patients             | ● In the context of the COVID-19 pandemic                                                                                |
| [110]       |   2023 | Healthcare Resources | ● Potential Application in Healthcare ● Automatic Notification and Prompt Action ● Web Server Integration                |

C. Li et al.

Table 4 A comparison of research on personalized medicine.

| Reference   |   Year | Target                                            | Advantage (s)                                                                                                                       |
|-------------|--------|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| [15]        |   2023 | Customized treatment based on patient populations | ● Fostering shared decision- making                                                                                                 |
| [111]       |   2022 | Personalized diabetes management                  | ● Focus on Personalized Diabetes Management ● Contribution to Prognosis Improvement                                                 |
| [112]       |   2022 | Individualized dietary recommendations            | ● IoT-based mHealth Monitoring System ● Consideration of Physical Activity ● Targeted at Children with                              |
| [113]       |   2022 | Personalised dental prediction and treatment      | Type 1 Diabetes ● incorporating personalized medicine in dental care                                                                |
| [114]       |   2023 | Personalized diabetes care                        | ● Utilization of 5G technology for real-time monitoring of diabetic patients ' health. ● Personalized care based on real-time data. |
| [115]       |   2023 | Personalized monitoring                           | ● The development of a hybrid algorithm for optimal feature selection                                                               |
| [116]       |   2022 | Personalized monitoring                           | ● Enabling quick adaptation of the well-generalized model                                                                           |

sensitivity with fast response/recovery. In addition, the proposed sensor exhibits great stability in the monitoring of breathing. It can provide an accurate recording of the breathing rate and depth of the patient in each respiratory pattern.

## 2.2. Communications

Communication technologies such as Bluetooth, Wi-Fi, and cellular networks are crucial for enabling data transfer between IoT devices and healthcare providers. These technologies help establish a seamless and real-time  communication  network  between  patients,  healthcare  pro­ viders, and medical equipment. Communication technologies such as Bluetooth, Wi-Fi, and cellular networks are instrumental in the seamless integration of IoT devices within healthcare. Bluetooth enables closerange,  low-power  connections,  perfect  for  wearable  health  gadgets, allowing real-time monitoring and data transmission to patient

G

Router smartphones [7]. Wi-Fi facilitates efficient data sharing within health­ care  facilities,  enabling  the  transfer  of  electronic  health  records  and medical images. Cellular networks [26], including 4 G and 5 G, offer broad  coverage  for  remote  patient  monitoring  and  telemedicine, ensuring healthcare providers have access to vital patient data regard­ less of location. Together, these technologies form a dynamic network that supports real-time communication, fosters timely decision-making, and enhances patient care. However, data security and privacy remain paramount considerations in their implementation.

There are several types of IoT communication technologies that are commonly used in healthcare, including:

## 2.2.1. Wi-Fi

Wireless internet connectivity is commonly used to connect IoT de­ vices to the internet and allow data to be transmitted across short dis­ tances, as shown in Fig. 1. It was first presented in 1997 [27] and has since  become  one  of  the  most  widely  used  wireless  communication technologies, providing trustworthy and fast connectivity to billions of devices  globally  [28].  This  technology  operates  on  the  IEEE  802.11 standard [29], which provides the technical specifications for wireless local area networks (WLANs). Wi-Fi allows multiple devices to connect and communicate with each other.

In recent years, Wi-Fi technology has continued to evolve, with the introduction of new standards such as Wi-Fi 6 and Wi-Fi 6E [30]. These standards provide faster speeds, corrected security, and increased ca­ pacity, allowing the advancement of new applications and services that require high-speed connectivity and low latency. As Wi-Fi continues to develop and improve, it is expected to play an increasingly important role in the medical industry and other sectors that require reliable and fast wireless connectivity[31].

## 2.2.2. Bluetooth

The name "Bluetooth" was inspired by Harald Bluetooth, a Viking king who unified Denmark and Norway in the 10th century. The name symbolizes  the  idea  of  uniting  different  devices  through  wireless communication. Developed by the Swedish telecommunications com­ pany Ericsson in 1994, this wireless communication technology is often used to connect IoT devices and smartphones or other mobile devices. Bluetooth  technology  uses  the  frequency-hopping  spread  spectrum (FHSS) [32] to avoid interference from other wireless devices operating in the same frequency range. The technology operates in the unlicensed industrial, scientific, and medical frequency bands, typically within the

Fig. 1. Data transmitted across short distances through Wi-Fi.

<!-- image -->

Internet C. Li et al.

range of 2.4 - 2.4835 GHz [33].

## 2.2.3. Zigbee

Zigbee is a low-power wireless communication protocol normally used for low data rate and long battery life IoT applications, such as remote patient monitoring devices. Zigbee was developed in the early 21st century [34], and the first version of the standard was released in 2004  [35].  Zigbee  uses  a  mesh  network  topology,  which  gives.  Key features of Zigbee include low power consumption, support for mesh networking, a focus on low data rates, and a cost-effective approach. It ' s known for its ability to enable devices to communicate wirelessly while conserving energy, making it well-suited for battery-powered IoT de­ vices. This is achieved through the use of sleep modes, in which devices can turn off their radios and consume very little power when they are not actively sending or receiving data. Zigbee also uses a collision avoidance technique called CSMA-CA to reduce the likelihood of data collisions [36].

## 2.2.4. Near Field Communication (NFC)

Near  Field  Communication  (NFC)  [37]  is  a  short-range  wireless communication technology that allows devices to communicate with each other over a distance of a few centimeters, as shown in Fig. 2. NFC is  based on Radio Frequency Identification (RFID) technology, which uses electromagnetic fields to transfer data wirelessly between two de­ vices.  NFC  technology  enables  a  wide  range  of  applications,  such  as contactless payments, access control, and data exchange between de­ vices [38].

## 2.3. Cloud

While  the  cloud  itself  is  not  a  communication  technology,  Cloud [39 - 41]  provides  a  platform  for  storing  [42],  managing[43],  and analyzing colossal amounts of healthcare data generated by IoT devices [44], as shown in Fig. 3. It has revolutionized the way businesses and individuals access and use computing resources, making it easier and more cost-effective to store and process large amounts of data.

The origins of cloud computing can be traced back to the 1960s [45], when early versions of utility computing were first developed. However, it wasn ' t until the late 1990s that the concept of cloud computing began to take shape [46].

In the context of healthcare IoT, by using cloud computing to store and  process  medical  data,  healthcare  providers  can  reduce  costs, improve patient outcomes, and enhance the overall quality of care. For example, cloud computing can be used to store electronic health records securely [47], making it easier for healthcare professionals to access and share patient data across different organizations and systems [43]. It can also be used to store and analyze large amounts of data from wearable devices and other medical IoT devices, allowing healthcare providers to detect and treat diseases more effectively. Additionally, cloud computing can provide real-time insights into patient health, enabling healthcare professionals to make more informed decisions and improve patient outcomes.

## 2.4. Artificial Intelligence (AI)

AI  algorithms  have  emerged  as  powerful  tools  in  healthcare, particularly when combined with data from IoT devices. By leveraging the vast amount of data these devices generate, AI algorithms can pro­ vide  real-time  insights  into  patient  health  and  enable  personalized healthcare interventions.

Machine  learning  models  [48],  a  subset  of  AI,  can  analyze  and identify patterns in IoT data to predict disease progression and suggest personalized treatment plans. These models can learn from historical data and detect subtle changes or anomalies in a patient ' s health pa­ rameters, allowing for early detection of potential health issues.

In the context of IoT in healthcare, AI algorithms can analyze data from wearable devices and other IoT sensors to monitor life functions, activity rates [49], sleep patterns [50], and other relevant metrics. By continuously  monitoring  and  analyzing  this  data,  AI  algorithms  can provide clinicians with early warnings of health deteriorations or re­ lapses, enabling timely interventions and improved patient outcomes.

Furthermore,  AI  algorithms  can  utilize  the  collected  IoT  data  to generate individualized care schemes [51]. By considering individual patient characteristics, medical history, and real-time data, these algo­ rithms can analyze large datasets to identify optimal treatment strate­ gies tailored to each patient ' s specific needs.

Incorporating the synergistic capabilities of AI and IoT will unveil a transformative  healthcare  model  poised  with  improved  streamlining and a steadfast focus on patient-centric methodologies. This evolution is characterized  by  marked  advancements  in  predictive  preciseness, underpinned by a nuanced, personalized approach to therapeutic stra­ tegies,  signifying  a  monumental  shift  towards  a  future  of  healthcare imbued with technological innovation and customized care pathways.

## 2.5. Blockchain

The  combination  of  blockchain  technology  [52 - 54]  and  IoT  in healthcare holds tremendous potential for enhancing data security and facilitating  seamless  data  sharing  among  healthcare  providers.  Via harnessing the blockchain ' s decentralized and immutable nature [55], patient data can be safeguarded against unauthorized access, tampering, and breaches. Through its distributed ledger system, blockchain ensures that  each  transaction  or  data  entry  is  cryptographically  secured  and linked to previous records, creating an unalterable chain, as shown in Fig.  4.  This  not  only  preserves  the  integrity  of  patient  data  but  also provides transparency and accountability in data management.

With blockchain, healthcare professionals can securely access and exchange  relevant  patient  information,  regardless  of  geographical  or organizational  boundaries.  This  streamlined  data  sharing  promotes better  care  coordination,  as  it  allows  healthcare  providers  to  have  a C. Li et al.

Fig. 2. NFC architecture.

<!-- image -->

Fig. 3. Cloud computing in IoT.

<!-- image -->

Fig. 4. How blockchain works.

<!-- image -->

comprehensive view of a patient ' s medical history and make informed decisions.  Patients  can  have  more  control  over  their  own  data  by granting  permission  for  specific  healthcare  providers  to  access  their information. The use of smart contracts, a self-executing code on the blockchain, automates data-sharing agreements and ensures compliance with  privacy  regulations,  giving  patients  greater  transparency  and control over their data.

## 2.6. Edge computing

Edge computing [56,57] plays a pivotal role in IoT in healthcare, offering significant advantages in real-time data processing  and improved performance.

In the context of IoT in healthcare, edge computing addresses the challenges  posed  by  network  latency  and  the  need  for  fast  response times [58]. Traditionally, IoT devices would transmit data to centralized cloud servers for processing and analysis, which can introduce delays in critical  healthcare applications. However, with edge computing, data processing occurs at the edge of the network, near the data source itself.

One of the key benefits of edge computing in healthcare is reduced network latency [59]. By processing data locally on IoT devices, edge computing minimizes the time required for data to travel back and forth between  the  device  and  a  distant  cloud  server  [60].  This  enables real-time  or  near-real-time  analysis  of  healthcare  data,  crucial  in time-sensitive  applications  such  as  remote  patient  monitoring,  emer­ gency response systems, and continuous health monitoring [61].

Furthermore, the introduction of edge computing significantly bol­ sters the integrity and confidentiality of data within IoT healthcare in­ frastructures.  By ensuring that sensitive health-related information is retained and processed directly on the devices, edge computing mini­ mizes the vulnerability to unauthorized intrusions and potential data compromise. In the realm of healthcare, where the sanctity of data is held  in  the  highest  regard,  such  an  enhancement  is  instrumental  in fortifying  the  resilience  of  information  systems  against  breaches, thereby safeguarding patient data with reinforced security protocols. This approach underscores a meticulous consideration of privacy, sub­ stantiating  a  robust  framework  where  data  security  is  meticulously upheld.

Additionally,  edge  computing  enables  offline  operation  and  resil­ ience to network failures [62]. In scenarios where internet connectivity is intermittent or unavailable, edge devices can continue to function and process  data  locally,  ensuring  uninterrupted  healthcare  services  and critical operations.

C. Li et al.

## 2.7. 5G

In the context of IoT in healthcare, 5G networks enable real-time communication [63], allowing healthcare providers to monitor remotely [64] and interact with patients in a continuous manner. The enhanced speed and low latency of 5G networks ensure that data can be transmitted  and received  almost instantaneously, enabling timely re­ sponses to critical healthcare situations.

One of the key advantages of 5G networks is their ability to support a vast  number  of  connected  devices  [65].  This  scalability  is  crucial  in healthcare settings where numerous IoT devices are interconnected. The high capacity of 5G networks ensures that a large volume of data can be transmitted simultaneously, enabling healthcare providers to efficiently collect and analyze real-time patient data.

The reliability of 5G networks [66] ensures uninterrupted connec­ tivity, which is essential in healthcare applications where continuous monitoring and real-time communication are critical. This reliability is particularly beneficial in remote areas or during emergency situations where access to healthcare services may be limited. With 5G networks, healthcare providers can deliver high-quality care remotely and bridge the gap between patients and medical experts. The reduced delay in data transmission ensures that healthcare providers can access and analyze patient  information  promptly,  enabling  timely  diagnoses,  treatment adjustments, and interventions.

Furthermore,  5G  networks  provide  the  foundation  for  advanced technologies  in  healthcare,  such  as  augmented  reality  (AR)  [67,68], virtual reality (VR) [69,70], and telepresence [71,72]. These technolo­ gies  have  the  potential  to  revolutionize  medical  education  [73,74], surgical training [75,76], and remote patient care [77,78] by enabling immersive  experiences  and  remote  collaboration  among  healthcare professionals.

## 3. IoT in different healthcare settings

## 3.1. Indoor healthcare environments

In indoor healthcare environments such as hospitals, IoT devices can be used to monitor patients, manage medical equipment, and track in­ ventory.  IoT  sensors  can  also  monitor  environmental  factors  such  as temperature,  humidity,  and  air  quality,  which  can  help  prevent  the spread of infections.

In indoor healthcare environments, IoT offers helpful functions for tracking the location of patients and healthcare staff. Through utilizing wearable  devices,  RFID  tags  [79],  or  smartphone-based  solutions, real-time monitoring of individuals ' movements becomes possible. This technology  facilitates  efficient  navigation,  optimizes  workflow,  and enhances emergency response times, leading to improved patient care. This  paper  [80]  proposes  a  deep  CNN-LSTM  architecture  for  indoor location estimation using a Passive Infrared (PIR) sensor in IoT health­ care systems. This paper [81] focuses on the use of an IoT model to analyze audience behavior in a resort, specifically emphasizing the in­ door system for tracking individuals ' movements and identifying their behavioral patterns.

Furthermore, indoor location technology enables the tracking and management of medical equipment and assets. By attaching RFID tags or utilizing sensor networks, healthcare providers can efficiently monitor medical devices ' location, usage, and maintenance needs, ensuring their availability when needed and reducing costs associated with equipment loss or misplacement. This research [82] proposes a non-intrusive in­ door  location  framework  using  wireless  sensor  networks  for  IoT healthcare. Through blockchain-based smart contracts, it implements role-based access control for system security.

IoT-based  indoor  localization  systems  also  contribute  to  environ­ mental  monitoring  in  healthcare  facilities.  By  deploying  sensor  net­ works,  parameters  such  as  temperature,  humidity,  air  quality,  and occupancy can be monitored in real-time [83]. This enables automated adjustments to ensure a comfortable and safe environment for patients and staff while also promoting energy efficiency and reducing the risk of healthcare-associated infections. This research [84] focuses on an in­ door system for IoT healthcare using visible light  communication. It investigates a MIMO VLC system that monitors indoor environments, collects data, and employs different modulation techniques for reliable communication. It offers an efficient solution for monitoring  various data types in IoT healthcare applications.

In terms of security and emergency response, IoT in indoor health­ care environments facilitates the implementation of advanced surveil­ lance  systems,  access  control,  and  emergency  call  systems.  These technologies enhance security measures, prevent unauthorized access, and enable swift responses to emergencies, ensuring the safety of pa­ tients, staff, and assets within the facility.

In hospitals, IoT devices such as wearable sensors and smart beds [85] can continuously monitor patient health, alerting healthcare pro­ viders to changes in vital signs or patient activity. This can help reduce the risk of adverse events and improve patient outcomes.

## 3.2. Outdoor healthcare environments

In addition to personal localization and remote patient monitoring, the integration of smart and IoT-enabled facilities in outdoor healthcare environments  plays  a  pivotal  role  in  enhancing  the  well-being  of individuals.

Public spaces such as parks, recreational areas, and transportation hubs  are  now  equipped  with  advanced  healthcare  amenities  that leverage  IoT  technology  to  provide  seamless  healthcare  access  [86]. These smart facilities are designed to meet the diverse healthcare needs of the public efficiently.

One notable aspect of smart outdoor healthcare environments is the utilization of IoT-enabled kiosks. These interactive kiosks serve as in­ formation hubs, offering real-time data on nearby healthcare providers, medication  dispensaries,  and  emergency  contact  numbers  [87].  In­ dividuals can access personalized health information, locate the nearest healthcare  facilities,  and  receive  immediate  assistance  in  case  of emergencies.

Furthermore, it is anticipated that in the future, smart parks and recreational areas could be equipped with fitness stations featuring IoTconnected  exercise  equipment  [88].  These  envisioned  smart  fitness stations  would  have the  potential  to  monitor  users ' activity,  provide personalized exercise recommendations, and track vital health metrics. It is envisioned that users may access these anticipated facilities through smartphone apps, potentially allowing for a tailored fitness experience based on individual health goals and preferences.

Transportation hubs also benefit from IoT integration. Automated external defibrillators (AEDs) [89] are now equipped with IoT sensors that continuously monitor their functionality. In case of any issues or malfunctions, automatic alerts are sent to healthcare providers, ensuring that these life-saving devices are always ready for use.

In  summary,  outdoor  healthcare  environments  are  evolving  to embrace smart and IoT-enabled facilities that enhance the healthcare experience in public spaces. From interactive kiosks to connected fitness equipment and advanced navigation systems, these technologies ensure that  healthcare  resources  are  readily  accessible,  personalized,  and responsive to the unique needs of individuals in outdoor settings.

## 3.3. Remote patient monitoring

IoT devices such as wearables and remote monitoring systems can be used to monitor patient health outside of traditional healthcare settings. Remote  patient  monitoring  can  help  improve  patient  outcomes  by detecting changes in health status early and enabling healthcare pro­ viders to intervene quickly [90].

Remote  Patient  Monitoring(RPM)  utilizes  technology,  such  as wearable devices,  sensors,  and  telecommunication  tools,  to  remotely C. Li et al.

monitor patients ' health status, collect relevant health data, and facili­ tate  virtual  consultations  between  patients  and  healthcare  providers. This approach allows for ongoing monitoring of vital signs, medication adherence, symptoms, and other health indicators without the need for frequent in-person visits.

One  of  the  significant  benefits  of  RPM  is  its  ability  to  provide healthcare  services  to  patients  in  remote  or  underserved  areas  [91]. Patients  living  in  rural  or  geographically  isolated  locations  may  face challenges accessing specialized healthcare services. RPM bridges this gap by enabling remote  monitoring  and  consultations, allowing healthcare providers to deliver timely interventions and guidance.

Moreover, RPM offers a convenient solution for patients with limited mobility, such as those with physical disabilities or chronic conditions that make travel difficult. By leveraging remote monitoring technolo­ gies,  patients  can  receive  the  necessary  healthcare  support  from  the comfort of their own homes. Healthcare providers can remotely assess their condition, adjust treatment plans, and provide guidance without the need for frequent hospital visits.

For patients with paralysis or other mobility limitations, RPM pro­ vides a lifeline to healthcare. They can use wearable devices [92], such as smartwatches or wireless sensors, to track their health parameters, including  heart  rate,  blood  pressure,  glucose  levels,  and  more.  The collected data is transmitted securely to healthcare professionals who can analyze it and provide appropriate medical interventions.

By  implementing  RPM,  healthcare  providers  can  ensure  regular monitoring and early detection of health issues, reducing the risk of complications  and  hospital  readmissions.  This  proactive  approach  to care management improves patient outcomes and enhances the overall quality of life for patients with limited mobility.

## 3.4. Smart cities

IoT  technologies  can  be  used  to  create  smart  cities  that  support healthier  lifestyles.  Smart  cities  can  incorporate  IoT  devices  such  as smart traffic lights, air quality monitors, and noise sensors, which can help reduce the risk of respiratory illness, noise pollution, and accidents.

Governments  have  recognized  the  potential  benefits  of  RPM  in improving healthcare outcomes, reducing healthcare costs, and enhancing patient satisfaction. As a result, they have taken various ac­ tions to support and advance the adoption of RPM in healthcare systems.

Governments  have  been  actively  involved  in  establishing  robust policies and regulations that prioritize patient data privacy and security in the context of RPM [93]. They enforce compliance with data pro­ tection laws, develop guidelines for secure data transmission and stor­ age, and promote the use of encryption and authentication measures to safeguard patient information.

Additionally, governments allocate substantial funding and financial support to healthcare organizations to facilitate the implementation and integration of RPM into existing healthcare systems. They also invest in infrastructure development, such as broadband connectivity and tele­ communication  networks,  to  ensure  remote  monitoring  services  are accessible even in remote or underserved areas. Moreover, governments actively foster collaborations and partnerships among healthcare pro­ viders, technology companies, and research institutions to drive inno­ vation, share best practices, and develop standardized protocols for RPM [94]. These collective efforts demonstrate governments ' commitment to creating  an  environment  that  supports  RPM  adoption,  improves healthcare delivery, and ultimately enhances patient outcomes.

For  individuals,  smart  cities  offer  personalized  healthcare  experi­ ences through technologies like telemedicine and wearable health de­ vices. These tools provide remote access to medical professionals and real-time health monitoring, empowering  individuals to actively manage their well-being. Additionally, smart cities facilitate convenient access to medical services, making healthcare more accessible.

On a public scale, smart cities contribute to better population health through data-driven insights. They can detect and mitigate environmental factors affecting public health, such as air pollution and disease outbreaks. Smart urban planning promotes active lifestyles with green spaces and efficient transportation systems. Moreover, these cities incorporate public health monitoring and control mechanisms. They can proactively track and manage health risks in public spaces, including early warning systems for potential disease outbreaks and real-time air quality  alerts.  Ultimately,  smart  cities  create  a  healthier  and  more accessible healthcare ecosystem that benefits both individuals and the broader public, improving overall well-being.

## 4. IoT applications in healthcare

## 4.1. Remote monitoring

In the dynamic landscape of healthcare technology, the integration of the IoT into remote health monitoring represents a pivotal advance­ ment, setting a novel paradigm in patient care and medical data man­ agement. Recent models leverage IoT to craft a nexus where portable physiological  monitoring  seamlessly  intertwines  with  sophisticated expert systems [95], thus enabling an ecosystem that fosters accuracy, immediacy,  and  reliability  in  diagnosis  and  patient  care.  Navigating through  the  latest  literature  unveils  a  spectrum  of  innovations  that collectively contribute to refining the reliability and efficiency of remote health  monitoring  systems.  These  developments  are  instrumental  in driving forward the boundaries of technological integration in health­ care, catalyzing a phase marked by unprecedented precision and effec­ tiveness in patient monitoring and care.

This study [96] presents a remote long-suffering monitoring system using IoT and cloud computing for accurate and constant health moni­ toring, abnormality detection, and alarm reports. The system includes a portable function for data visualization, loading, and real-time remote monitoring. It has been proven to measure biological parameters and tested for reliable performance in real-time data transmission.

The research [97] emphasizes the monitoring of remote patients ' body temperature using embedded infrared sensors and WiFi connec­ tivity  in  an  IoT-based  system.  The  prototype  demonstrates  accurate temperature  measurement  and  offers  guidance  on  the  optimal  body parts for effective temperature monitoring.

The proposal [98] suggests a secure and private integration of IoT with healthcare units to achieve a reliable and secure remote patient monitoring  system.  It  utilizes  secure  authentication,  communication, and privacy measures, including a smartwatch, server, and smartphone application for monitoring vital signs.

ML-based  algorithms  [99]  enhance  patient  health  surveillance through  remote  monitoring,  efficient  data  mining,  and  improved healthcare services.

This  study  [100]  presents  a  convenient  electrocardiogram  moni­ toring system that links patients and doctors through a cloud server. It allows remote monitoring, investigation, and alarms, reducing hospital appointments.  Experimental  results  demonstrate  trustworthiness  and comparisons  with  existing  techniques,  followed  by  a  system  perfor­ mance demonstration.

The  paper  [101]  proposes  using  IoT-based  remote  patient  moni­ toring  with  medical  IoT  wearables  and  ML  algorithms  to  monitor Post-Covid patients and provide immediate medical assistance.

This paper [102] presents the implementation of a private backend server software for an IoT health monitoring system. The system uploads data to a cloud server after detecting vital signs. Compared to existing patient  monitoring systems, the system demonstrates impressive per­ formance  and  accuracy,  achieving  a  high  success  rate  for  IoT  data broadcasts.

The proposed framework in this study [103] utilizes a Convolutional Neural Network with Long-Short Term Memory (CNN-LSTM) to create a Patient Activity Monitoring System for accurate human action recog­ nition. The results demonstrate a high accuracy rate of 99.53 %, sur­ passing existing algorithms by at least 4.73 %.

C. Li et al.

This research [104] proposes a smart system using IoT technology to assist patients with influenza symptoms in determining their COVID-19 infection status. The system enables quick medical care and tracking of patient mobility to analyze suspected behaviors, with potential appli­ cations in medical institutions, quarantine units, and airports.

## 4.2. Locating systems

Through real-time locating systems (RTLS), healthcare providers can quickly locate the position of patients, particularly in emergency and critical situations, enabling timely medical assistance and reducing the risk  of  patient  harm.  The  research  on  RTLS  extends  beyond  patient location  and  encompasses  accurate  and  up-to-date  information  on healthcare resource locations. By implementing RTLS, healthcare facil­ ities can effectively manage and track vital medical resources such as equipment, medications, and supplies, ensuring their timely availability, reducing search time, and preventing inventory shortages or delays in patient care.

This study [105] proposes an approach that leverages deep neural networks and spatial-spectrum fitting to achieve precise and effective source localization within UAV networks with radar arrays. The method enhances both the accuracy and real-time processing capabilities.

This study [106] utilizes RFID and NB-IoT technology to precisely track and monitor medical resources in the context of specimen logistics in a hospital. By integrating IoT capabilities such as real-time location tracking and environmental condition monitoring, the system enhances the  efficiency  and  reliability  of  specimen  transport.  It  effectively  ad­ dresses the challenges associated with the logistics and management of medical resources.

This work [107] proposes a wearable device system suitable for in­ dividuals who are prone to getting lost, such as those with Alzheimer ' s disease. The system enables location tracking and step detection both indoors and outdoors. Data is documented and accessed through a user web application, allowing for anomaly detection, patient positioning, and warnings to caregivers. The system aims to enhance independence and provide peace of mind to caregivers.

The study [108] demonstrates the use of sensor-agnostic APIs and ontology  modules  to  support  complex  functionalities  and  address environment-wellness relationships. It focuses on location-based appli­ cations, such as environmental sensing and wellness monitoring, and proposes an ontology-based approach.

This research [8] tracks critical health parameters in the direct. Data is conveyed to the cloud storing system and displayed to practitioners and  affected  roles  through  a  mobile  application.  The  system  allows timely medical decisions based on abnormal values. The study [109] implemented  similar  functionality  in  the  context  of  the  COVID-19 pandemic.

The IoT-enabled embedded device [110], integrated with health care systems, monitors the trash levels in public bins. This IoT solution en­ sures timely waste management, reducing the city ' s risk of unsanitary conditions and potential health hazards.

The IoT-enabled embedded device, integrated with health care sys­ tems, not only monitors the trash levels in public bins but also holds the potential for future applications in managing medical waste in hospitals. By leveraging IoT technology, hospitals can implement similar moni­ toring systems for medical waste disposal. This would ensure efficient management of biomedical waste, reduce the risk of contamination, and maintain  a  hygienic  environment  within  healthcare  facilities.  The adoption of IoT in hospital waste management can improve infection control practices and promote the safety and well-being of patients and healthcare workers.

## 4.3. Personalized medicine

In the realm of healthcare, the concept of Personalized Medicine has emerged as a groundbreaking frontier. This approach tailors medical treatment to the unique genetic, environmental, and lifestyle factors of individual patients, holding the promise of more effective and precise healthcare. To gain a deeper understanding of the current state of this evolving field, we delve into a comprehensive study that explores the latest advancements and trends in personalized medicine.

This study [15] validates the relationship between IoT adoption and patient care service engagement. The findings highlight the importance of personalized medicine and the need for a reliable computing envi­ ronment to ensure privacy and security. The study provides a framework for healthcare providers to empower patients and improve the patient care delivery environment, fostering shared decision-making and supe­ rior care services.

This paper [111] focuses on utilizing the Latent Dirichlet allocation (LDA) model in the context of personalized diabetes management. The aim is to predict future treatment approaches. This emphasizes the po­ tential of personalized diabetes management to enhance patient prog­ noses and signifies a noteworthy advancement in healthcare decision-making.

The target group of this study [112] is children with type 1 diabetes. The system considers the effect of activity on the extent of the patient ' s illness, allowing doctors to provide individualized dietary recommen­ dations based on predicted changes in blood glucose levels. This study emphasizes the potential of IoT and mHealth technologies in delivering personalized healthcare solutions for pediatric diabetes management.

This  paper  [113]  presents  a  personalized  dental  caries  prediction model using machine learning. This personalized approach eliminates the hard detection process and enables dentists and patients to assess the risk of future severe dental infections.

This project [114] utilizes 5G technology to monitor diabetic pa­ tients ' health in real-time, providing early alerts and personalized care.

This  study  [115]  introduces  a  system  that  utilizes  exhaled  VOC analysis for personalized monitoring and disease diagnosis. It provides various benefits, including early diagnosis, cost reduction, and improved quality of care.

This  study  [116]  proposes  a  meta-learning-based  personalization approach for  individualized  health  monitoring,  which  realizes  better accuracy than accessible approaches in detecting vital signs and human activity.

## 5. Discussion

IoT has brought tremendous advancements to healthcare, empow­ ering healthcare providers, patients, and systems alike. By harnessing the potential of IoT technology, the healthcare industry can improve patient outcomes, optimize resource utilization, and transform health­ care delivery into a more personalized and proactive approach. How­ ever, it is imperative to address security and privacy concerns to ensure IoT ' s responsible and ethical implementation in healthcare.

Medical IoT devices collect and transmit a large amount of sensitive information, including personal health data, medical history, and bio­ metric measurements. In research settings, scholars must establish clear and  transparent  privacy  policies  and  consent  mechanisms  to  inform patients about the collected data, how it will be used, and with whom it will be shared. Patients should have control over their data and be able to provide informed consent for its usage. Anonymization techniques such  as  de-identification  [117]  or  differential  privacy  [118]  can  be employed  to  protect  patient  privacy  further  while  allowing  for  data analysis and research. This involves implementing robust authentication and  access  control  mechanisms  to  ensure  that  only  authorized  in­ dividuals can access the devices and data. Encryption techniques should be employed to protect data during transmission and storage, preventing interception  or  tampering.  Regular  security  audits  and  vulnerability assessments are necessary to identify and address any weaknesses or vulnerabilities in the system.

This  is  an  area  that  many  scholars  are  currently  dedicated  to researching.  Progress  has  been  made  in  the  development  of  security C. Li et al.

protocols, encryption algorithms, and privacy-enhancing technologies specific  to  the  healthcare  domain.  The  paper  [119]  introduces  a healthcare  architecture  that  utilizes  blockchain  technology  to  tackle data management and security concerns and identify unauthorized ac­ tivities in the communication chain. The experimental findings reveal a significant success rate in detecting hostile IoT devices.

This study [120] proposes a new key exchange protocol for health­ care applications. The protocol utilizes secure algorithms to ensure data security without compromising performance. It effectively blocks key exchange attacks and realizes improved performance.

This  research  [121]  introduces  Medi-Block,  a  secure  healthcare system combining Blockchain and IoT. It enables remote patient moni­ toring  and  provides  an  additional  security  layer  for  personalized healthcare. The framework ensures privacy and maximizes the value of healthcare.

For  healthcare  providers,  it  is  crucial  to  establish  robust  security protocols, including data encryption, access control mechanisms, and secure communication channels [122], to prevent unauthorized access and data breaches. Respecting patient privacy rights helps build trust within healthcare organizations and maintains ethical standards.

The healthcare IoT ecosystem faces unique challenges in ensuring security  and  privacy.  It  involves  various  devices  and  systems  with different  security  functionalities  and  vulnerabilities.  Healthcare  pro­ viders must deal with the complexity of integrating and protecting these devices  while  complying  with  regulatory  requirements,  such  as  the Health Insurance Portability and Accountability Act (HIPAA) [123] in the  United  States  or  similar  data  protection  laws  in  other  coun­ tries/regions. The interoperability of different IoT devices and platforms also poses challenges in maintaining consistent security levels throughout the ecosystem.

Establishing  clear  and  transparent  privacy  policies  and  consent mechanisms or legislation is crucial at the government level. Govern­ ments  should  develop  privacy  protection  laws  and  regulations  for healthcare  IoT,  clearly  defining  data  collection,  usage,  and  sharing principles. Additionally, appropriate regulatory bodies should  be established  to  oversee  and  enforce  privacy  protection  policies  while imposing penalties for violations of privacy regulations.

Governments and relevant institutions should also provide education and training to enhance the privacy awareness and skills of healthcare professionals, enabling them to properly handle and protect sensitive data. Furthermore, public education initiatives should be launched to raise awareness among individuals about privacy protection.

In  the  realm  of  interoperability,  aligning  various  IoT  devices  and platforms to uphold consistent security benchmarks across the ecosystem presents another layer of complexity. Therefore, a cohesive strategy  that  encapsulates  comprehensive  legislation,  clear  privacy policies, and robust enforcement mechanisms is essential to navigate the multifaceted security challenges intrinsic to the healthcare IoT landscape.

## 6. Optimizing IoT applications in healthcare: key strategies and implementation measures

Optimizing IoT applications in healthcare requires the implementa­ tion of key strategies and measures to ensure the effective and efficient utilization  of  this  technology.  By  adopting  the following  approaches, healthcare organizations can maximize the benefits of IoT and enhance patient care:

## 6.1. Seamless integration and interoperability

One of the key strategies is to ensure the seamless integration of IoT devices and systems within the existing healthcare infrastructure. This involves developing standardized protocols and interfaces [116] that facilitate  interoperability  between  different  IoT  devices,  electronic health  records  (EHR)  systems,  and  other  healthcare  applications.  By enabling  smooth  data  exchange  and  communication,  healthcare  pro­ viders can leverage the full potential of IoT to enhance care coordination and improve patient outcomes.

## 6.2. Robust security and privacy measures

Identity  and  access  management  serve  as  critical  components  in reducing security risks [124]. Multifactor authentication and stringent access controls are pivotal in bolstering security measures[125]. These protocols guarantee that only authorized individuals can access patient data, reducing the risk of unauthorized breaches. Additionally, it is vital not  to  overlook  the  security  of  the  IoT  devices  themselves.  Device manufacturers must integrate security features, such as secure boot and hardware-based security modules [126]. Consistent patching and firm­ ware updates are also essential to counter emerging security vulnera­ bilities  and  threats.  Data  privacy  represents  yet  another  essential dimension. Techniques like data anonymization and de-identification protect  patient  privacy  while  still  permitting  valuable  data  analysis [127].  The  incorporation  of  differential  privacy  methods  further  en­ hances privacy by introducing controlled noise into datasets. Healthcare organizations must place high priority on compliance with industry-specific regulations. These  regulations establish stringent standards for patient data security and privacy, underlining the legal aspects of data protection. Promoting user training and awareness is a crucial  endeavor.  Informed  healthcare  staff  and  patients  actively contribute to threat detection and reporting. Moreover, a well-defined incident  response  plan  is  imperative.  This  plan  should  delineate  the necessary steps to take in the event of a security breach, assigning spe­ cific  responsibilities  to  individuals  or  teams  for  each  action.  Lastly, third-party vendor assessment is a pivotal facet of healthcare IoT secu­ rity. Rigorously evaluating the security practices of third-party vendors and service providers ensures their alignment with stringent security standards and protocols.

## 6.3. Data analytics and real-time insights

IoT generates vast amounts of data that, when properly analyzed, can provide valuable insights for healthcare decision-making. By leveraging advanced analytics techniques, such as machine learning and artificial intelligence, healthcare providers can gain real-time insights from IoTgenerated  data.  These  insights  can  support  early  detection  of  health conditions,  proactive  intervention,  and  personalized  treatment  plans, ultimately improving patient outcomes and operational efficiency.

## 6.4. Scalability and flexibility

As  IoT  technologies  continue  to  evolve,  healthcare  organizations must ensure scalability and flexibility in their implementation. This in­ volves designing IoT infrastructure that can accommodate future growth and the integration of new devices and applications. Scalable cloudbased platforms and architectures provide the necessary flexibility to adapt  to  changing  healthcare  demands,  allowing  for  the  seamless addition and management of IoT devices and data streams.

## 6.5. Government support and strategic location selection

Government  involvement  plays  a  crucial  role  in  the  successful implementation  of  IoT  in  healthcare.  Governments  can  support  the establishment of smart hospitals by providing incentives, funding, and regulatory frameworks that encourage the adoption of IoT technologies. Additionally, the selection of suitable locations for smart hospitals is important. Governments can support the establishment of smart hospi­ tals by providing incentives, funding, and regulatory frameworks that encourage the adoption of IoT technologies. Factors such as proximity to healthcare facilities, availability of robust connectivity infrastructure, and accessibility to a skilled workforce should be considered to ensure C. Li et al.

the optimal functioning of IoT-enabled healthcare systems.

By implementing these key strategies and measures, healthcare or­ ganizations can optimize IoT applications in healthcare and unlock their full potential. This includes improving patient care, enhancing opera­ tional efficiency, and advancing healthcare research and innovation. It is vital for organizations to stay updated on technological advancements, collaborate with industry partners, and continuously evaluate and refine their IoT strategies to align with the evolving needs of the healthcare landscape.

## 7. Conclusion

This manuscript furnishes an exhaustive analysis of the intersection between  the  Internet  of  Things  (IoT)  and  healthcare,  delineating  its transformative potential within the sector. Central to the discourse is the burgeoning  relevance  of  personalized  medicine,  underscored  by  a framework of standardized protocols and rigorous regulations pivotal in safeguarding  privacy  and  bolstering  security  infrastructures.  Current incorporations of IoT technologies within healthcare paradigms have heralded enhanced patient care trajectories and improved clinical out­ comes, heralding innovations in remote patient monitoring paradigms, individualized treatment architectures, and operational efficiencies in healthcare delivery frameworks.

However, it is imperative to recognize the spectrum of challenges that punctuate the seamless integration of IoT within healthcare eco­ systems. These necessitate a symbiotic engagement amongst healthcare professionals, regulatory architects, and the broader research commu­ nity  to  foster  solutions  that  are  both  innovative  and  pragmatically viable.

While this contribution endeavors to provide nuanced insights and seminal contributions within the realm of healthcare-focused IoT, it is cognizant of the existence of additional seminal works and innovative strides within this dynamic field that may not have been encapsulated within the present discourse. In this vein, we extend our apologies for any  inadvertent  omissions.  In  summation,  this  manuscript  aspires  to augment the collaborative exploration of leveraging IoT technologies toward  engendering  transformative  enhancements  within  healthcare landscapes and auxiliary domains.

## Funding

This  study  is  partially  supported  by  British  Heart  Foundation Accelerator Award, UK (AA/18/3/34220); Hope Foundation for Cancer Research, UK (RM60G0680); Medical Research Council Confidence in Concept Award, UK (MC\_PC\_17171); Biotechnology and Biological Sci­ ences Research Council, UK (RM32G0178B8).

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data Availability

The authors do not have permission to share data.

## References

- [1] [V.A. Dang, et al., Intelligent healthcare: integration of emerging technologies and internet of things for humanity, Sensors 23 (9) (2023) 4200.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref1)
- [2] [M. Mansour, et al., Internet of things: a comprehensive overview on protocols, architectures, technologies, simulation tools, and future directions, Energies 16 (8) (2023) 3465.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref2)
- [3] [Z. Haitaamar, et al., Lower inclination orbit concept for direct-communication-tosatellite internet-of-things using lean satellite standard in near-equatorial regions, Appl. Sci. 13 (9) (2023) 5654.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref3)
- [4] [D. Shehada, et al., Fog-based distributed trust and reputation management system for internet of things, J. King Saud. Univ. - Comput. Inf. Sci. 34 (10) (2022) 8637 - 8646.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref4)
- [5] [B. Tekinerdogan, ¨ O. K ¨ oksal, T. Çelik, System architecture design of IoT-based smart cities, Appl. Sci. 13 (7) (2023) 4173.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref5)
- [6] [M. Ryalat, H. ElMoaqet, M. AlFaouri, Design of a smart factory based on cyberphysical systems and internet of things towards industry 4.0, Appl. Sci. 13 (4) (2023) 2156.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref6)
- [7] [H.H. Alshammari, The internet of things healthcare monitoring system based on MQTT protocol, Alex. Eng. J. 69 (2023) 275 - 287.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref7)
- [8] [B.G. Mohammed, D.S. Hasan, Smart healthcare monitoring system using IoT, iJIM 17 (01) (2023) 141.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref8)
- [9] [S. Tiwari, K. Nahak, A. Mishra, Revolutionizing healthcare: the power of IoT in health monitoring, J. Data Acquis. Process. 38 (2) (2023) 2416.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref9)
- [10] [S. Krishnamoorthy, A. Dua, S. Gupta, Role of emerging technologies in future IoTdriven Healthcare 4.0 technologies: a survey, current challenges and future directions, J. Ambient Intell. Humaniz. Comput. 14 (1) (2023) 361 - 407.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref10)
- [11] [M. Juma, F. Alattar, B. Touqan, Securing big data integrity for industrial IoT in smart manufacturing based on the trusted consortium blockchain (TCB), IoT 4 (1) (2023) 27 - 55.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref11)
- [12] [R. Krishankumar, F. Ecer, Selection of IoT service provider for sustainable transport using q-rung orthopair fuzzy CRADIS and unknown weights, Appl. Soft Comput. 132 (2023), 109870.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref12)

[13]

[H. Jiang, et al., RETRACTED ARTICLE: creating a ubiquitous learning](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref13)

[environment using IoT in transportation, Soft Comput. 27 (2) (2023), 1213-1213.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref13)

- [14] [H. Zeng, et al., An IoT and blockchain-based approach for the smart water management system in agriculture, Expert Syst. 40 (4) (2023), e12892.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref14)
- [15] [V. Bhatt, S. Chakraborty, Improving service engagement in healthcare through internet of things based healthcare systems, J. Sci. Technol. Policy Manag. 14 (1) (2023) 53 - 73.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref15)
- [16] [M.B. Alazzam, F. Alassery, A. Almulihi, Federated deep learning approaches for the privacy and security of IoT systems, Wirel. Commun. Mob. Comput. 2022 (2022) 1 - 7.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref16)
- [17] [S.S. Albouq, et al., A survey of interoperability challenges and solutions for dealing with them in IoT environment, IEEE Access 10 (2022) 36416 - 36428.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref17)
- [18] [J. Shahid, et al., Data protection and privacy of the internet of healthcare things (IoHTs), Appl. Sci. 12 (4) (2022) 1927.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref18)
- [19] [T. Taft, et al., Are we there yet?" Ten persistent hazards and inefficiencies with the use of medication administration technology from the perspective of practicing nurses. J. Am. Med. Inform. Assoc. 30 (5) (2023) 809 - 818.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref19)
- [20] [E.A. Banu, V. Rajamani, Design of online vitals monitor by integrating big data and IoT, Comput. Syst. Sci. Eng. 44 (3) (2023) 2469 - 2487.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref20)
- [21] [R. Gunawan, A. Andang, M. Ridwan, Performance comparison for hearth rate signal detection for different location in fingertip and wrist using sensor MAX30102, J. Biomim. Biomater. Biomed. Eng. 59 (2023) 131 - 143.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref21)
- [22] [Y. Kim, et al., Self-powered wearable micropyramid piezoelectric film sensor for real-time monitoring of blood pressure, Adv. Eng. Mater. 25 (2) (2023), 2200873.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref22)
- [23] [K. Qin, W. Huang, T. Zhang, Multitask deep label distribution learning for blood pressure prediction, Inf. Fusion 95 (2023) 426 - 445.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref23)
- [24] [A.G. Arifah, A.N. Costrada, H. Harmadi, Inovasi pulse oximeter dengan sumber cahaya LED Merah dan Inframerah Yang Dilengkapi Suhu Tubuh Menggunakan sensor MLX90614 Berbasis IoT, J. Fis. Unand 12 (2) (2023) 199 - 205.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref24)
- [25] [Z. Song, et al., High-sensitivity paper-based capacitive humidity sensors for respiratory monitoring, IEEE Sens. J. 23 (3) (2023) 2291 - 2302.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref25)
- [26] [K. Mehmood, K. Kralevska, D. Palma, Intent-driven autonomous network and service management in future cellular networks: a structured literature review, Comput. Netw. 220 (2023), 109477.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref26)

[27]

[S. Dongus, et al., Health effects of WiFi radiation: a review based on systematic](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref27)

[quality evaluation, Crit. Rev. Environ. Sci. Technol. 52 (19) (2022) 3547](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref27)

[-](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref27)

[3566.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref27)

- [28] [L.C. Paul, et al., A dual-band semi-circular patch antenna for WiMAX and WiFi-5/ 6 applications, Int. J. Commun. Syst. 36 (1) (2023), e5357.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref28)
- [29] [H. Taramit, L. Orozco-Barbosa, A. Haqiq, A renewal theory based performance and configuration framework of the IEEE 802.11 ah RAW mechanism, Digit. Commun. Netw. 9 (1) (2023) 236 - 251.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref29)
- [30] [H.Y. Jiang, et al., A wideband circularly polarized dielectric patch antenna with a modified air cavity for Wi-Fi 6 and Wi-Fi 6E applications, IEEE Antennas Wirel. Propag. Lett. 22 (1) (2023) 213 - 217.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref30)
- [31] [A.S. George, A.H. George, T. Baskar, Wi-Fi 7: the next frontier in wireless connectivity, Partners Univ. Int. Innov. J. 1 (4) (2023) 133 - 145.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref31)
- [32] [J. Ye, H. Gharavi, B. Hu, Fast beam discovery and adaptive transmission under frequency selective attenuations in Sub-Terahertz bands, IEEE Trans. Signal Process. 71 (2023) 727 - 740.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref32)
- [33] [C.-Y. Chen, et al., Using iBeacon components to design and fabricate low-energy and simple indoor positioning method, Sens. Mater. 35 (3) (2023) 703 - 722.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref33)
- [34] [X. Cheng, Y. Fan, Research and design of intelligent speech equipment in smart english language lab based on Internet of Things technology, Procedia Comput. Sci. 198 (2022) 505 - 511.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref34)
- [35] [V.A. Orfanos, et al., A comprehensive review of IoT networking technologies for smart home automation applications, J. Sens. Actuator Netw. 12 (2) (2023) 30.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref35)
- [36] [R. Tanash, M. AlQudah, S. Al-Agtash, Enhancing energy efficiency of IEEE 802.15. 4-based industrial wireless sensor networks, J. Ind. Inf. Integr. 33 (2023), 100460.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref36)
- [37] [W.-W. Yu, C.-Y. Fang, The role of near-field communication mobile payments in sustainable restaurant operations: a restaurateur ' s perspective, Sustainability 15 (16) (2023) 12471.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref37)

C. Li et al.

- [38] [S.H. Ahammad, et al., Defected ground - structured symmetric circular ring antenna for near-field scanning plasmonic applications, Plasmonics 18 (2) (2023) 541 - 549.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref38)
- [39] [F.A. AlSelami, Major cloud computing security challenges with innovative approaches, Tehnicki Glas.-Technical J. 17 (1) (2023) 141 - 145.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref39)
- [40] [A.R. Khan, L.K. Alnwihel, A brief review on cloud computing authentication frameworks, Eng. Technol. Appl. Sci. Res. 13 (1) (2023) 9997 - 10004.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref40)
- [41] [S.S. Pericherla, Cloud computing threats, vulnerabilities and countermeasures: a state-of-the-Art . ISECURE-ISC, Int. J. Inf. Security 15 (1) (2023) 1 - 58.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref41)
- [42] [J. Logeshwaran, G. Ramesh, V. Aravindarajan, A secured database monitoring method to improve data backup and recovery operations in cloud computing, BOHR Int. J. Comput. Sci. 2 (1) (2023) 1 - 7.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref42)
- [43] [D.K. Sharma, et al., The aspect of vast data management problem in healthcare sector and implementation of cloud computing technique, Mater. Today.: Proc. 80 (2023) 3805 - 3810.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref43)
- [44] [A.A. Khamis, et al., Development and performance evaluation of an iot-integrated breath analyzer, Int. J. Environ. Res. Public Health 20 (2) (2023) 1319.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref44)
- [45] [A. Raja Santhi, P. Muthuswamy, Industry 5.0 or industry 4.0 S? Introduction to industry 4.0 and a peek into the prospective industry 5.0 technologies, Int. J. Interact. Des. Manuf. (IJIDeM) 17 (2) (2023) 947 - 979.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref45)
- [46] [A. Gupta, et al., Role of cloud computing in management and education, Mater. Today.: Proc. 80 (2023) 3726 - 3729.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref46)
- [47] [P. Sharma, et al., EHDHE: enhancing security of healthcare documents in IoTenabled digital healthcare ecosystems using blockchain, Inf. Sci. 629 (2023) 703 - 718.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref47)
- [48] [Z. Chang, et al., Landslide susceptibility prediction using slope unit-based machine learning models considering the heterogeneity of conditioning factors, J. Rock. Mech. Geotech. Eng. 15 (5) (2023) 1127 - 1143.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref48)
- [49] [F.S. Russell-Pavier, et al., A highly scalable and autonomous spectroscopic radiation mapping system with resilient IoT detector units for dosimetry, safety and security, J. Radiol. Prot. 43 (1) (2023), 011503.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref49)
- [50] [B.J. Dakhale, et al., An automatic sleep-scoring system in elderly women with osteoporosis fractures using frequency localized finite orthogonal quadrature Fejer Korovkin kernels, Med. Eng. Phys. 112 (2023), 103956.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref50)
- [51] [M.A.I. Mozumder, et al., Metaverse for digital anti-aging healthcare: an overview of potential use cases based on artificial intelligence, blockchain, IoT technologies, its challenges, and future directions, Appl. Sci. 13 (8) (2023) 5127.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref51)
- [52] [S. Khan, et al., Investigating the barriers of blockchain technology integrated food supply chain: a BWM approach, Benchmark.: Int. J. 30 (3) (2023) 713 - 735.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref52)
- [53] [M. Kouhizadeh, Q. Zhu, J. Sarkis, Circular economy performance measurements and blockchain technology: an examination of relationships, Int. J. Logist. Manag. 34 (3) (2023) 720 - 743.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref53)
- [54] [F. Tao, Y.-Y. Wang, S.-H. Zhu, Impact of blockchain technology on the optimal pricing and quality decisions of platform supply chains, Int. J. Prod. Res. 61 (11) (2023) 3670 - 3684.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref54)
- [55] [F.A. Reegu, et al., Blockchain-based framework for interoperable electronic health records for an improved healthcare system, Sustainability 15 (8) (2023) 6337.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref55)
- [56] [F. Al-Doghman, et al., AI-enabled secure microservices in edge computing: opportunities and challenges, IEEE Trans. Serv. Comput. 16 (2) (2023) 1485 - 1504.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref56)
- [57] [T. Kim, et al., MoDEMS: optimizing edge computing migrations for user mobility, IEEE J. Sel. Areas Commun. 41 (3) (2023) 675 - 689.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref57)
- [58] [A. Bourechak, et al., At the confluence of artificial intelligence and edge computing in iot-based applications: a review and new perspectives, Sensors 23 (3) (2023) 1639.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref58)
- [59] [H. Hua, et al., Edge computing with artificial intelligence: a machine learning perspective, ACM Comput. Surv. 55 (9) (2023) 1 - 35.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref59)
- [60] [W. Xu, et al., Edge learning for B5G networks with distributed signal processing: Semantic communication, edge computing, and wireless sensing, IEEE J. Sel. Top. Signal Process. 17 (1) (2023) 9 - 39.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref60)
- [61] [Q. Cheng, et al., A real-time UAV target detection algorithm based on edge computing, Drones 7 (2) (2023) 95.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref61)
- [62] [M. Raeisi-Varzaneh, et al., Resource scheduling in edge computing: architecture, taxonomy, open issues and future research directions, IEEE Access 11 (2023) 25329 - 25350.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref62)
- [63] [A. Lagorio, et al., 5G in Logistics 4.0: potential applications and challenges, Procedia Comput. Sci. 217 (2023) 650 - 659.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref63)
- [64] [T.A. Suleiman, A. Adinoyi, Telemedicine and smart healthcare - the role of artificial intelligence, 5G, cloud services, and other enabling technologies, Int. J. Commun., Netw. Syst. Sci. 16 (3) (2023) 31 - 51.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref64)
- [65] [A. Mughaid, et al., Improved dropping attacks detecting system in 5g networks using machine learning and deep learning approaches, Multimed. Tools Appl. 82 (9) (2023) 13973 - 13995.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref65)
- [66] [M. Attaran, The impact of 5G on the evolution of intelligent automation and industry digitization, J. Ambient Intell. Humaniz. Comput. 14 (5) (2023) 5977 - 5993.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref66)
- [67] [J.R. Lex, et al., Clinical applications of augmented reality in orthopaedic surgery: a comprehensive narrative review, Int. Orthop. 47 (2) (2023) 375 - 391.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref67)
- [68] [A. Sufyan, et al., From 5G to beyond 5G: a comprehensive survey of wireless network evolution, challenges, and promising technologies, Electronics 12 (10) (2023) 2200.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref68)
- [69] [A. Parmaxi, Virtual reality in language learning: a systematic review and implications for research and practice, Interact. Learn. Environ. 31 (1) (2023) 172 - 184.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref69)
- [70] [Z. Huang, et al., Standard evolution of 5G-advanced and future mobile network for extended reality and metaverse, IEEE Internet Things Mag. 6 (1) (2023) 20 - 25.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref70)
- [71] [L.E. Johannessen, E.B. Rasmussen, M. Haldar, Student at a distance: exploring the potential and prerequisites of using telepresence robots in schools, Oxf. Rev. Educ. 49 (2) (2023) 153 - 170.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref71)
- [72] [G. de Medeiros Sousa, A.M.P. Santos, The viability of telesurgery service in the Autonomous Region of the Azores, supported by the 5G network, Procedia Comput. Sci. 219 (2023) 422 - 430.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref72)
- [73] [T.N. Fitria, Augmented Reality (AR) and Virtual Reality (VR) technology in education: media of teaching and learning: a review, Int. J. Comput. Inf. Syst. (IJCIS) 4 (1) (2023) 14 - 25.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref73)
- [74] [A.S. Ahuja, et al., The digital metaverse: applications in artificial intelligence, medical education, and integrative health, Integr. Med. Res. 12 (1) (2023), 100917.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref74)
- [75] [J. Mitchell, et al., Surgical training for civilian surgeons interested in humanitarian surgery: a scoping review, J. Surg. Res. 283 (2023) 282 - 287.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref75)
- [76] [D.T. Guerrero, et al., Advancing surgical education: the use of artificial intelligence in surgical training, Am. Surg. 89 (1) (2023) 49 - 54.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref76)
- [77] [A. Sadhu, et al., A review of data management and visualization techniques for structural health monitoring using BIM and virtual or augmented reality, J. Struct. Eng. 149 (1) (2023) 03122006.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref77)
- [78] [H. Kwon, et al., Review of smart hospital services in real healthcare environments, Healthc. Inform. Res. 28 (1) (2022) 3 - 15.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref78)
- [79] [W. Wang, et al., Zero-power screen printed flexible RFID sensors for smart home, J. Ambient Intell. Humaniz. Comput. 14 (4) (2023) 3995 - 4004.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref79)
- [80] [K. Ngamakeur, et al., Deep CNN-LSTM network for indoor location estimation using analog signals of passive infrared sensors, IEEE Internet Things J. 9 (22) (2022) 22582 - 22594.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref80)
- [81] [R. Ferinia, et al., Factors determining customers desire to analyse supply chain management in intelligent IoT, J. Comb. Optim. 45 (2) (2023) 72.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref81)
- [82] [A. To ˇ si ´ c, N. Hrovatin, J. Vi ˇ ci ˇ c, A WSN framework for privacy aware indoor location, Appl. Sci. 12 (6) (2022) 3204.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref82)
- [83] [J. Oh, et al., Indoor environmental quality improvement in green building: Occupant perception and behavioral impact, J. Build. Eng. 69 (2023), 106314.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref83)
- [84] [M.A.S. Sejan, W.-Y. Chung, Performance analysis of a long-range MIMO VLC system for indoor IoT, IEEE Internet Things J. 10 (8) (2022) 6999 - 7010.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref84)
- [85] [M. Maddeh, et al., Spatio-temporal cluster mapping system in smart beds for patient monitoring, Sensors 23 (10) (2023) 4614.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref85)
- [86] [M.N.U. Nabi, F.T. Zohora, S. Misbauddin, Social media links with social capital to trust in healthcare facilities: empirical evidence from Bangladesh, Libr. Hi Tech. 41 (1) (2023) 210 - 228.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref86)
- [87] [C. Masquillier, et al., Development and implementation of a community health literacy hub, ' Health Kiosk '- a grassroots innovation, Front. Public Health 10 (2023) 1069255.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref87)
- [88] [M.J. Kang, Y.C. Hwang, Exploring the factors affecting the continued usage intention of IoT-based healthcare wearable devices using the TAM model, Sustainability 14 (19) (2022) 12492.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref88)
- [89] [E. Berglund, et al., Effect of smartphone dispatch of volunteer responders on automated external defibrillators and out-of-hospital cardiac arrests: the SAMBA randomized clinical trial, JAMA Cardiol. 8 (1) (2023) 81 - 88.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref89)
- [90] [T. Shaik, et al., Remote patient monitoring using artificial intelligence: current state, applications, and challenges, Wiley Interdiscip. Rev.: Data Min. Knowl. Discov. 13 (2) (2023), e1485.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref90)
- [91] [N. Mohammadzadeh, S. Rezayi, S. Saeedi, Telemedicine for patient management in remote areas and underserved populations, Disaster Med. Public Health Prep. 17 (2023), e167.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref91)
- [92] [J.-J. Wang, R. Payne, A survey of internet of things in healthcare, EAI Endorsed Trans. Internet Things 7 (27) (2022), e4.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref92)
- [93] [L. Xia, D. Semirumi, R. Rezaei, A thorough examination of smart city applications: exploring challenges and solutions throughout the life cycle with emphasis on safeguarding citizen privacy, Sustain. Cities Soc. 98 (2023), 104771.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref93)
- [94] [H.S. Bednar, et al., Building global health systems capacity during COVID-19 to improve vaccination access and reduce hesitancy: case studies in Zambia and Tanzania, Health Secur. 21 (5) (2023) 341 - 346.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref94)
- [95] [P. Valsalan, et al., Remote healthcare monitoring using expert system, Int. J. Adv. Comput. Sci. Appl. 13 (3) (2022) 593 - 599.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref95)
- [96] [M.L. Sahu, et al., Cloud-based remote patient monitoring system with abnormality detection and alert notification, Mob. Netw. Appl. 27 (5) (2022) 1894 - 1909.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref96)
- [97] [W. Boonsong, N. Senajit, Remote patient body temperature monitoring basedIEEE802.11a internet of things (IoT), Prz. Elektrotech. 98 (6) (2022) 95 - 98.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref97)
- [98] [M.I. Ahmed, G. Kannan, Secure and lightweight privacy preserving Internet of things integration for remote patient monitoring, J. King Saud Univ.-Comput. Inf. Sci. 34 (9) (2022) 6895 - 6908.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref98)
- [99] [S. Deepa, et al., IoT-enabled smart healthcare data and health monitoring based machine learning algorithms, J. Intell. Fuzzy Syst. 44 (2) (2023) 2927 - 2941.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref99)
- [100] [A. Bushnag, A wireless ECG monitoring and analysis system using the IoT cloud, Intell. Autom. Soft Comput. 33 (1) (2022) 51 - 70.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref100)
- [101] [S. Rahman, et al., Post-covid remote patient monitoring using medical internet of things and machine learning analytics, Scalable Comput. - Pract. Experience 24 (1) (2023) 1 - 16.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref101)
- [102] [F.I. Ali, T.E. Ali, Z.T. Al-dahan, Private backend server software-based telehealthcare tracking and monitoring system, Int. J. Online Biomed. Eng. 19 (1) (2023) 119 - 134.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref102)

C. Li et al.

- [103] [K. Balasubramanian, et al., A hybrid deep learning for patient activity recognition (PAR): Real time body wearable sensor network from healthcare monitoring system (HMS), in: Journal of Intelligent &amp; Fuzzy Systems, 44, 2023, pp. 195 - 211.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref103)
- [104] [T.A. Abdulhussein, et al., Early coronavirus disease detection using internet of things smart system, Int. J. Electr. Comput. Eng. 13 (1) (2023) 1161 - 1168.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref104)
- [105] [J.Y. Cong, et al., CRB weighted source localization method based on deep neural networks in multi-UAV network, IEEE Internet Things J. 10 (7) (2023) 5747 - 5759.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref105)
- [106] [N.T. Le, et al., Deployment of smart specimen transport system using RFID and NB-IoT technologies for hospital laboratory, Sensors 23 (1) (2023) 546.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref106)
- [107] [A. García-Requejo, et al., Activity monitoring and location sensory system for people with mild cognitive impairments, IEEE Sens. J. 23 (5) (2023) 5448 - 5458.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref107)
- [108] M. Zappatore, et al., Semantic models for IoT sensing to infer environment - wellness relationships, Future Gener. Comput. Syst. 140 (2023) 1 - 17.
- [109] [R. Yoganapriya, P. Deepthi, M. Dhinakaran, IoT based covid patient health monitoring system, Int. J. Eng. Technol. Manag. Sci. 7 (1) (2023) 82 - 87.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref109)
- [110] [M. Karthik, et al., An efficient waste management technique with IoT based smart garbage system, Mater. Today.: Proc. 80 (2023) 3140 - 3143.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref110)
- [111] [C. Ni Ki, et al., Topic modelling in precision medicine with its applications in personalized diabetes management, Expert Syst. 39 (4) (2022), e12774.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref111)
- [112] [N. Zholdas, et al., A personalized mHealth monitoring system for children and adolescents with T1 diabetes by utilizing IoT sensors and assessing physical activities, Int. J. Comput. Commun. Control 17 (2022) 3.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref112)
- [113] I.-A. Kang, et al., DCP: prediction of dental caries using machine learning in personalized medicine, Appl. Sci. 12 (2022), https://doi.org/10.3390/ app12063043.
- [114] [M.V.S. Sravani, et al., 5G-smart diabetes: toward personalized diabetes diagnosis with healthcare big data clouds, Turk. J. Comput. Math. Educ. (Turcomat) 14 (2) (2023) 99 - 106.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref114)
- [115] [G. Karthick, P. Pankajavalli, Chronic obstructive pulmonary disease prediction using Internet of things-spiro system and fuzzy-based quantum neural network classifier, Theor. Comput. Sci. 941 (2023) 55 - 76.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref115)
- [116] [Z. Jia, Y. Shi, J. Hu, Personalized neural network for patient-specific health monitoring in IoT: a metalearning approach, IEEE Trans. Comput. - Aided Des. Integr. Circuits Syst. 41 (12) (2022) 5394 - 5407.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref116)
- [117] [Y. S ¸ ahin, ˙ I. DOGRU, An enterprise data privacy governance model: securitycentric multi-model data anonymization, Int. J. Eng. Res. Dev. 15 (2) (2023) 574 - 583.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref117)
- [118] [J. Vasa, A. Thakkar, Deep learning: differential privacy preservation in the era of big data, J. Comput. Inf. Syst. 63 (3) (2023) 608 - 631.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref118)
- [119] [A.I. Taloba, et al., A blockchain-based hybrid platform for multimedia data processing in IoT-Healthcare, Alex. Eng. J. 65 (2023) 263 - 274.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref119)
- [120] [M. Al-Zubaidie, Implication of lightweight and robust hash function to support key exchange in health sensor networks, Symmetry 15 (1) (2023) 152.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref120)
- [121] Bhatt, V. and S. Chakraborty, MediBlock: a pervasive way to create healthcare value in secured manner for personalized care, in Recent Advances in Blockchain Technology: Real-World Applications, S.K. Panda, et al., Editors. 2023, Springer International Publishing: Cham. p. 233 - 243.
- [122] [O.A. Garcia Valencia, et al., Ethical implications of chatbot utilization in nephrology, J. Pers. Med. 13 (9) (2023) 1363.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref121)
- [123] [B. Krzyzanowski, S.M. Manson, Twenty years of the health insurance portability and accountability act safe harbor provision: unsolved challenges and ways forward, JMIR Med. Inform. 10 (8) (2022), e37756.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref122)
- [124] [A.S. George, S. Sagayarajan, Securing cloud application infrastructure: understanding the penetration testing challenges of IaaS, PaaS, and SaaS environments, Partn. Univers. Int. Res. J. 2 (1) (2023) 24 - 34.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref123)
- [125] [S. Ahmed, M. Khan, Securing the Internet of Things (IoT): a comprehensive study on the intersection of cybersecurity, privacy, and connectivity in the IoT ecosystem, AI, IoT Fourth Ind. Revolut. Rev. 13 (9) (2023) 1 - 17.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref124)
- [126] [A.J. Cabrera-Guti ´ errez, et al., Secure sensor prototype using hardware security modules and trusted execution environments in a blockchain application: wine logistic use case, Electronics 12 (13) (2023) 2987.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref125)
- [127] [S. Sampaio, et al., Collecting, processing and secondary using personal and (Pseudo) anonymized data in smart cities, Appl. Sci. 13 (6) (2023) 3830.](http://refhub.elsevier.com/S0925-2312(23)01140-2/sbref126)