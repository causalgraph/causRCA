graph [
  directed 1
  multigraph 1
  node [
    id 19
    label "Hyd_IsEnabled"
    type "['Variable']"
    iri "cg_store#Hyd_IsEnabled"
    comment "['True = indicates that hydraulics is enabled']"
    groupedBy "Hydraulics"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['Hyd_A_700204_Disables_Hyd_IsEnabled', 'Hyd_A_700207_Disables_Hyd_IsEnabled']"
  ]
  node [
    id 20
    label "Hyd_Pressure"
    type "['Variable']"
    iri "cg_store#Hyd_Pressure"
    comment "['hydraulic pressure analog ACTUAL value']"
    groupedBy "Hydraulics"
    io "input"
    datatype "int"
    variabletype "continuous"
    examples "10274, 7381"
    isCausing "['Hyd_Pressure_Monitoring', 'Hyd_Pressure_Control']"
  ]
  node [
    id 21
    label "Hyd_Valve_P_Up"
    type "['Variable']"
    iri "cg_store#Hyd_Valve_P_Up"
    comment "['True = accumulator charge valve commanded to open (until target pressure is reached again)']"
    groupedBy "Hydraulics"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['Hyd_Pressure_Control', 'Hyd_A_700202_Disables_Hyd_Valve_P_Up']"
  ]
  node [
    id 23
    label "Hyd_Filter_Ok"
    type "['Variable']"
    iri "cg_store#Hyd_Filter_Ok"
    comment "['True = hydraulic filter is Ok']"
    groupedBy "Hyd_Filter"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Hyd_Filter_Monitoring', 'Hyd_Filter_Clogged_Pump_Failure']"
  ]
  node [
    id 26
    label "Hyd_Level_Ok"
    type "['Variable']"
    iri "cg_store#Hyd_Level_Ok"
    comment "['True = hydraulic level is Ok']"
    groupedBy "Hyd_Level"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Hyd_Level_Warning_Monitoring', 'Hyd_Level_Alarm_Monitoring', 'Hyd_Level_Low_Pump_Failure']"
  ]
  node [
    id 28
    label "Hyd_Pump_isOff"
    type "['Variable']"
    iri "cg_store#Hyd_Pump_isOff"
    comment "['True = hydraulic pump is off']"
    groupedBy "Hyd_Pump"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['Hyd_Pump_On_Status']"
  ]
  node [
    id 29
    label "Hyd_Pump_Ok"
    type "['Variable']"
    iri "cg_store#Hyd_Pump_Ok"
    comment "['True = motor of hydraulic pump is Ok']"
    groupedBy "Hyd_Pump"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Hyd_Pump_Monitoring', 'Hyd_Pump_Ok_Condition']"
    isAffectedBy "['Hyd_Filter_Clogged_Pump_Failure', 'Hyd_Level_Low_Pump_Failure']"
  ]
  node [
    id 30
    label "Hyd_Pump_On"
    type "['Variable']"
    iri "cg_store#Hyd_Pump_On"
    comment "['True = hydraulic pump is commanded to be on']"
    groupedBy "Hyd_Pump"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Hyd_Pump_On_Status']"
    isAffectedBy "['Hyd_Pump_Ok_Condition', 'Hyd_A_700202_Disables_Hyd_Pump_On', 'Hyd_A_700206_Disables_Hyd_Pump_On']"
  ]
  node [
    id 33
    label "Hyd_Temp_lt_70"
    type "['Variable']"
    iri "cg_store#Hyd_Temp_lt_70"
    comment "['True = temperature of hydraulic oil below 70 degree C, False = warning for temperature']"
    groupedBy "Hyd_Temp"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Hyd_Temp_Warning_Monitoring', 'Hyd_Temp_70_to_80']"
  ]
  node [
    id 34
    label "Hyd_Temp_lt_80"
    type "['Variable']"
    iri "cg_store#Hyd_Temp_lt_80"
    comment "['True = temperature of hydraulic oil below 80 degree C, False = system stop']"
    groupedBy "Hyd_Temp"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Hyd_Temp_Alarm_Monitoring']"
    isAffectedBy "['Hyd_Temp_70_to_80']"
  ]
  node [
    id 18
    label "Hyd_A_700202"
    type "['Alarm']"
    iri "cg_store#Hyd_A_700202"
    comment "['alarm that pressure monitoring of hydraulics registered hydraulic pressure out of norm']"
    groupedBy "Hydraulics"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isCausing "['Hyd_A_700202_Disables_Hyd_Valve_P_Up', 'Hyd_A_700202_Disables_Hyd_Pump_On']"
    isAffectedBy "['Hyd_Pressure_Monitoring']"
  ]
  node [
    id 22
    label "Hyd_A_700207"
    type "['Alarm']"
    iri "cg_store#Hyd_A_700207"
    comment "['alarm that hydraulic filter for returning oil is clogged']"
    groupedBy "Hyd_Filter"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isCausing "['Hyd_A_700207_Disables_Hyd_IsEnabled']"
    isAffectedBy "['Hyd_Filter_Monitoring']"
  ]
  node [
    id 24
    label "Hyd_A_700205"
    type "['Alarm']"
    iri "cg_store#Hyd_A_700205"
    comment "['warning that hydraulic oil level is close to minimum']"
    groupedBy "Hyd_Level"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['Hyd_Level_Warning_Monitoring']"
  ]
  node [
    id 25
    label "Hyd_A_700206"
    type "['Alarm']"
    iri "cg_store#Hyd_A_700206"
    comment "['alarm that hydraulic oil level is below minimum']"
    groupedBy "Hyd_Level"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isCausing "['Hyd_A_700206_Disables_Hyd_Pump_On']"
    isAffectedBy "['Hyd_Level_Alarm_Monitoring']"
  ]
  node [
    id 27
    label "Hyd_A_700208"
    type "['Alarm']"
    iri "cg_store#Hyd_A_700208"
    comment "['alarm that hydraulic pump motor protection switch triggered']"
    groupedBy "Hyd_Pump"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['Hyd_Pump_Monitoring']"
  ]
  node [
    id 31
    label "Hyd_A_700203"
    type "['Alarm']"
    iri "cg_store#Hyd_A_700203"
    comment "['warning that temperature monitoring of hydraulics registers more than 70 degree C']"
    groupedBy "Hyd_Temp"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['Hyd_Temp_Warning_Monitoring']"
  ]
  node [
    id 32
    label "Hyd_A_700204"
    type "['Alarm']"
    iri "cg_store#Hyd_A_700204"
    comment "['alarm that temperature monitoring of hydraulics registers more than 80 degree C']"
    groupedBy "Hyd_Temp"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isCausing "['Hyd_A_700204_Disables_Hyd_IsEnabled']"
    isAffectedBy "['Hyd_Temp_Alarm_Monitoring']"
  ]
  edge [
    source 20
    target 18
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Pressure_Monitoring"
    comment "['When hydraulic pressure is out of normal range for at least 5 seconds -> Alarm 700202 is triggered']"
    hasCause "Hyd_Pressure"
    hasEffect "Hyd_A_700202"
  ]
  edge [
    source 20
    target 21
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Pressure_Control"
    comment "['When hydraulic pressure falls below threshold, valve is commanded to open (Hyd_Valve_P_Up=True)']"
    hasCause "Hyd_Pressure"
    hasEffect "Hyd_Valve_P_Up"
  ]
  edge [
    source 23
    target 22
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Filter_Monitoring"
    comment "['When Hyd_Filter_Ok=False for at least 5 seconds -> Alarm 700207 is triggered']"
    hasCause "Hyd_Filter_Ok"
    hasEffect "Hyd_A_700207"
  ]
  edge [
    source 23
    target 29
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Filter_Clogged_Pump_Failure"
    comment "['When Hyd_Filter_Ok=False for some time, the probability of Hyd_Pump_OK=False rises (pump might fail)']"
    hasCause "Hyd_Filter_Ok"
    hasEffect "Hyd_Pump_Ok"
  ]
  edge [
    source 26
    target 24
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Level_Warning_Monitoring"
    comment "['When Hyd_Level_Ok=False -> Warning 700205 is immediately triggered']"
    hasCause "Hyd_Level_Ok"
    hasEffect "Hyd_A_700205"
  ]
  edge [
    source 26
    target 25
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Level_Alarm_Monitoring"
    comment "['When Hyd_Level_Ok=False for at least 800ms  -> Alarm 700206 is triggered']"
    hasCause "Hyd_Level_Ok"
    hasEffect "Hyd_A_700206"
  ]
  edge [
    source 26
    target 29
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Level_Low_Pump_Failure"
    comment "['When Hyd_Level_Ok=False for some time, the probability of Hyd_Pump_OK=False rises (pump might fail)']"
    hasCause "Hyd_Level_Ok"
    hasEffect "Hyd_Pump_Ok"
  ]
  edge [
    source 29
    target 27
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Pump_Monitoring"
    comment "['When Hyd_Pump_Ok=False -> Motor protection switch is triggered indicated by alarm 700208=True']"
    hasCause "Hyd_Pump_Ok"
    hasEffect "Hyd_A_700208"
  ]
  edge [
    source 29
    target 30
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Pump_Ok_Condition"
    comment "['If Hyd_Pump_Ok=False, then Hyd_Pump_On is turned off (Hyd_Pump_On=False) to prevent damage.']"
    hasCause "Hyd_Pump_Ok"
    hasEffect "Hyd_Pump_On"
  ]
  edge [
    source 30
    target 28
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Pump_On_Status"
    comment "['When hydraulic pump is commanded to be on (Hyd_Pump_On=True), it results in the pump being on (Hyd_Pump_isOff=False)']"
    hasCause "Hyd_Pump_On"
    hasEffect "Hyd_Pump_isOff"
  ]
  edge [
    source 33
    target 31
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Temp_Warning_Monitoring"
    comment "['When Hyd_Temp_lt_70=False (temp >= 70&#176;C) -> Warning 700203 is triggered']"
    hasCause "Hyd_Temp_lt_70"
    hasEffect "Hyd_A_700203"
  ]
  edge [
    source 33
    target 34
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Temp_70_to_80"
    comment "['Temperature above 70 degree is reached before 80 degree (common unobserved confounder)']"
    hasCause "Hyd_Temp_lt_70"
    hasEffect "Hyd_Temp_lt_80"
  ]
  edge [
    source 34
    target 32
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_Temp_Alarm_Monitoring"
    comment "['When Hyd_Temp_lt_80=False (temp >= 80&#176;C) -> Alarm 700204 is triggered']"
    hasCause "Hyd_Temp_lt_80"
    hasEffect "Hyd_A_700204"
  ]
  edge [
    source 18
    target 21
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700202_Disables_Hyd_Valve_P_Up"
    comment "['When hydraulic pressure out of norm alarm 700202 is triggered (Hyd_A_700202=True), the pressure accumulation valve is closed in fear of leakage (Hyd_Valve_P_Up=False).']"
    hasCause "Hyd_A_700202"
    hasEffect "Hyd_Valve_P_Up"
  ]
  edge [
    source 18
    target 30
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700202_Disables_Hyd_Pump_On"
    comment "['When pressure out of norm alarm 700202 is triggered (Hyd_A_700202=True), the hydraulic pump is turned off (Hyd_Pump_On=False).']"
    hasCause "Hyd_A_700202"
    hasEffect "Hyd_Pump_On"
  ]
  edge [
    source 22
    target 19
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700207_Disables_Hyd_IsEnabled"
    comment "['When hydraulic filter is clogged alarm 700207 is triggered (Hyd_A_700207=True), hydraulics is disabled (Hyd_IsEnabled=False).']"
    hasCause "Hyd_A_700207"
    hasEffect "Hyd_IsEnabled"
  ]
  edge [
    source 25
    target 30
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700206_Disables_Hyd_Pump_On"
    comment "['When hydraulic level is below minimum alarm 700206 is triggered (Hyd_A_700206=True), the hydraulic pump is turned off (Hyd_Pump_On=False).']"
    hasCause "Hyd_A_700206"
    hasEffect "Hyd_Pump_On"
  ]
  edge [
    source 32
    target 19
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700204_Disables_Hyd_IsEnabled"
    comment "['When hydraulic temperature is above 80 degree C alarm 700204 is triggered (Hyd_A_700204=True), hydraulics is disabled (Hyd_IsEnabled=False).']"
    hasCause "Hyd_A_700204"
    hasEffect "Hyd_IsEnabled"
  ]
]
