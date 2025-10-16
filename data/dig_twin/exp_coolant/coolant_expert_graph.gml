graph [
  directed 1
  multigraph 1
  node [
    id 2
    label "CLF_Filter_Ok"
    type "['Variable']"
    iri "cg_store#CLF_Filter_Ok"
    comment "['True = coolant filter for fine particles is in Ok state']"
    groupedBy "Filter"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['CLF_Filter_Monitoring']"
  ]
  node [
    id 4
    label "F_Filter_Ok"
    type "['Variable']"
    iri "cg_store#F_Filter_Ok"
    comment "['True = coolant fleece filter is Ok (not empty). False = Empty -> Fault']"
    groupedBy "Fleece"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Fleece_Filter_Monitoring']"
  ]
  node [
    id 6
    label "HP_Pump_isOff"
    type "['Variable']"
    iri "cg_store#HP_Pump_isOff"
    comment "['True = high pressure pump for coolant is turned off']"
    groupedBy "HighPressurePump"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['HP_Pump_not_ok_to_off', 'LP_Pump_On_HP_Pump_Off']"
  ]
  node [
    id 7
    label "HP_Pump_Ok"
    type "['Variable']"
    iri "cg_store#HP_Pump_Ok"
    comment "['True = high pressure pump for coolant is Ok']"
    groupedBy "HighPressurePump"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['HP_Pump_Monitoring', 'HP_Pump_not_ok_to_off']"
  ]
  node [
    id 9
    label "LT_Level_Ok"
    type "['Variable']"
    iri "cg_store#LT_Level_Ok"
    comment "['True = lifting tank level supervision is ok -> No overlow. False = overlflow']"
    groupedBy "LiftingTank"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['LT_Level_Monitoring', 'LT_Level_Pump_Ok']"
  ]
  node [
    id 10
    label "LT_Pump_Ok"
    type "['Variable']"
    iri "cg_store#LT_Pump_Ok"
    comment "['True = lifting tank pump supervision gives positive feedback']"
    groupedBy "LiftingTank"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['LT_Level_Pump_Ok']"
  ]
  node [
    id 12
    label "LP_Pump_Ok"
    type "['Variable']"
    iri "cg_store#LP_Pump_Ok"
    comment "['True = low pressure pump for coolant is Ok']"
    groupedBy "LowPressurePump"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['LP_Pump_Monitoring', 'LP_Pump_On_Condition']"
  ]
  node [
    id 13
    label "LP_Pump_On"
    type "['Variable']"
    iri "cg_store#LP_Pump_On"
    comment "['True = low pressure pump for coolant  is commanded to be on']"
    groupedBy "LowPressurePump"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['LP_Pump_On_HP_Pump_Off']"
    isAffectedBy "['LP_Pump_On_Condition']"
  ]
  node [
    id 15
    label "CLT_Level_lt_Min"
    type "['Variable']"
    iri "cg_store#CLT_Level_lt_Min"
    comment "['True = coolant level is below minimum -> Error']"
    groupedBy "Tank"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['CLT_Level_Min_Monitoring']"
  ]
  node [
    id 1
    label "CLF_A_700307"
    type "['Alarm']"
    iri "cg_store#CLF_A_700307"
    comment "['alarm that coolant filter is clogged']"
    groupedBy "Filter"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['CLF_Filter_Monitoring']"
  ]
  node [
    id 3
    label "F_A_700313"
    type "['Alarm']"
    iri "cg_store#F_A_700313"
    comment "['alarm that fleece supply in coolant is empty and needs replacement']"
    groupedBy "Fleece"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['Fleece_Filter_Monitoring']"
  ]
  node [
    id 5
    label "HP_A_700304"
    type "['Alarm']"
    iri "cg_store#HP_A_700304"
    comment "['alarm that coolant high pressure (HP) pump motor protection switch triggered']"
    groupedBy "HighPressurePump"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['HP_Pump_Monitoring']"
  ]
  node [
    id 8
    label "LT_A_700317"
    type "['Alarm']"
    iri "cg_store#LT_A_700317"
    comment "['alarm for overflow at lifting tank of coolant']"
    groupedBy "LiftingTank"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['LT_Level_Monitoring']"
  ]
  node [
    id 11
    label "LP_A_700301"
    type "['Alarm']"
    iri "cg_store#LP_A_700301"
    comment "['alarm that coolant low pressure (LP) pump motor protection switch triggered']"
    groupedBy "LowPressurePump"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['LP_Pump_Monitoring']"
  ]
  node [
    id 14
    label "CLT_A_700310"
    type "['Alarm']"
    iri "cg_store#CLT_A_700310"
    comment "['alarm that coolant tank fill level is below minimum']"
    groupedBy "Tank"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['CLT_Level_Min_Monitoring']"
  ]
  edge [
    source 2
    target 1
    key 0
    type "['CausalEdge']"
    iri "cg_store#CLF_Filter_Monitoring"
    comment "['When Filter_OK=False for at least 5 seconds -> Alarm 700307 is triggered']"
    hasCause "CLF_Filter_Ok"
    hasEffect "CLF_A_700307"
  ]
  edge [
    source 4
    target 3
    key 0
    type "['CausalEdge']"
    iri "cg_store#Fleece_Filter_Monitoring"
    comment "['When Fleece_Filter_Ok=False for at least 5 seconds --> Alarm 700313 is triggered']"
    hasCause "F_Filter_Ok"
    hasEffect "F_A_700313"
  ]
  edge [
    source 7
    target 5
    key 0
    type "['CausalEdge']"
    iri "cg_store#HP_Pump_Monitoring"
    comment "['When HP_Pump_Ok=False -> Alarm 700304 is triggered']"
    hasCause "HP_Pump_Ok"
    hasEffect "HP_A_700304"
  ]
  edge [
    source 7
    target 6
    key 0
    type "['CausalEdge']"
    iri "cg_store#HP_Pump_not_ok_to_off"
    comment "['When HP_Pump_Ok=False, then HP_Pump_isOff=True, i.e. the pump is turned off to prevent damage.']"
    hasCause "HP_Pump_Ok"
    hasEffect "HP_Pump_isOff"
  ]
  edge [
    source 9
    target 8
    key 0
    type "['CausalEdge']"
    iri "cg_store#LT_Level_Monitoring"
    comment "['When LT_Level_Ok=False -> Alarm 700317 is triggered']"
    hasCause "LT_Level_Ok"
    hasEffect "LT_A_700317"
  ]
  edge [
    source 9
    target 10
    key 0
    type "['CausalEdge']"
    iri "cg_store#LT_Level_Pump_Ok"
    comment "['When LT_Level_Ok=False -> LT_Pump_Ok=False, i.e. the pump is turned off to prevent overflow.']"
    hasCause "LT_Level_Ok"
    hasEffect "LT_Pump_Ok"
  ]
  edge [
    source 12
    target 11
    key 0
    type "['CausalEdge']"
    iri "cg_store#LP_Pump_Monitoring"
    comment "['When LP_Pump_Ok=False -> Alarm 700301 is triggered']"
    hasCause "LP_Pump_Ok"
    hasEffect "LP_A_700301"
  ]
  edge [
    source 12
    target 13
    key 0
    type "['CausalEdge']"
    iri "cg_store#LP_Pump_On_Condition"
    comment "['Only if LP_Pump_Ok=True, then LP_Pump_On can be True.']"
    hasCause "LP_Pump_Ok"
    hasEffect "LP_Pump_On"
  ]
  edge [
    source 13
    target 6
    key 0
    type "['CausalEdge']"
    iri "cg_store#LP_Pump_On_HP_Pump_Off"
    comment "['When LP Pump is commanded to be on, then HP Pump is turned off.']"
    hasCause "LP_Pump_On"
    hasEffect "HP_Pump_isOff"
  ]
  edge [
    source 15
    target 14
    key 0
    type "['CausalEdge']"
    iri "cg_store#CLT_Level_Min_Monitoring"
    comment "['When Level_lt_Min=True -> Alarm 700310 is triggered']"
    hasCause "CLT_Level_lt_Min"
    hasEffect "CLT_A_700310"
  ]
]
