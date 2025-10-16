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
    isAffectedBy "['LP_Pump_On_Condition', 'Spd_InnerCoolOn_LP_Pump_On']"
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
    id 16
    label "ExU_isOff"
    type "['Variable']"
    iri "cg_store#ExU_isOff"
    comment "['True = extraction unit is off (measured)']"
    groupedBy "ExtractionUnit"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "False, True"
    isAffectedBy "['ExU_On_ExU_isOff']"
  ]
  node [
    id 17
    label "ExU_On"
    type "['Variable']"
    iri "cg_store#ExU_On"
    comment "['True = extraction unit is commanded to turn on']"
    groupedBy "ExtractionUnit"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['ExU_On_ExU_isOff']"
  ]
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
    id 35
    label "Lubr_On"
    type "['Variable']"
    iri "cg_store#Lubr_On"
    comment "['True = lubrication is commanded to be on']"
    groupedBy "Lubrication"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Lubr_On_Lubr_P_Ok']"
  ]
  node [
    id 36
    label "Lubr_P_Ok"
    type "['Variable']"
    iri "cg_store#Lubr_P_Ok"
    comment "['True = pressure for central lubrication is sufficient; False = pressure needs to be built up']"
    groupedBy "Lubrication"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['Lubr_On_Lubr_P_Ok']"
  ]
  node [
    id 37
    label "MP_Inactive"
    type "['Variable']"
    iri "cg_store#MP_Inactive"
    comment "['True = measuring probe inactive, False=active and measuring']"
    groupedBy "MeasuringProbe"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "False, True"
    isAffectedBy "['MPA_WorkPos_MP_Inactive']"
  ]
  node [
    id 40
    label "MPA_InitPos"
    type "['Variable']"
    iri "cg_store#MPA_InitPos"
    comment "['True = axis that carries measuring probe is in initial position = hidden']"
    groupedBy "Axis"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['MPA_InitPos_Monitoring', 'MPA_InitPos_BothEndPosSensors', 'MPA_InitPos_MPC_close']"
    isAffectedBy "['MPA_toInitPos_MPA_InitPos']"
  ]
  node [
    id 41
    label "MPA_toInitPos"
    type "['Variable']"
    iri "cg_store#MPA_toInitPos"
    comment "['True = axis that carries measuring probe commanded to initial position (hidden)']"
    groupedBy "Axis"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['MPA_toInitPos_MPA_InitPos', 'MPA_toInitPos_MPA_toWorkPos', 'MPA_toInitPos_Monitoring']"
  ]
  node [
    id 42
    label "MPA_toWorkPos"
    type "['Variable']"
    iri "cg_store#MPA_toWorkPos"
    comment "['True = axis that carries measuring probe commanded to working position to measure workpieces']"
    groupedBy "Axis"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['MPA_toWorkPos_MPA_WorkPos', 'MPA_toWorkPos_Monitoring']"
    isAffectedBy "['MPA_toInitPos_MPA_toWorkPos', 'MPC_isOpen_MPA_toWorkPos']"
  ]
  node [
    id 43
    label "MPA_WorkPos"
    type "['Variable']"
    iri "cg_store#MPA_WorkPos"
    comment "['True = axis that carries measuring probe is in working pos, where it can measure workpieces']"
    groupedBy "Axis"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['MPA_WorkPos_MP_Inactive', 'MPA_WorkPos_Monitoring', 'MPA_WorkPos_BothEndPosSensors']"
    isAffectedBy "['MPA_toWorkPos_MPA_WorkPos']"
  ]
  node [
    id 44
    label "MPC_close"
    type "['Variable']"
    iri "cg_store#MPC_close"
    comment "['True = cover for measuring probe axis cover is commanded to be closed']"
    groupedBy "Cover"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['MPC_close_MPC_Closed', 'MPC_close_MPC_open']"
    isAffectedBy "['MPA_InitPos_MPC_close']"
  ]
  node [
    id 45
    label "MPC_Closed"
    type "['Variable']"
    iri "cg_store#MPC_Closed"
    comment "['True = cover for measuring probe axis cover is closed']"
    groupedBy "Cover"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['MPC_close_MPC_Closed']"
  ]
  node [
    id 46
    label "MPC_isOpen"
    type "['Variable']"
    iri "cg_store#MPC_isOpen"
    comment "['True = cover for measuring probe axis cover is open']"
    groupedBy "Cover"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['MPC_isOpen_MPA_toWorkPos']"
    isAffectedBy "['MPC_open_MPC_isOpen']"
  ]
  node [
    id 47
    label "MPC_open"
    type "['Variable']"
    iri "cg_store#MPC_open"
    comment "['True = cover for measuring probe axis cover is commanded to be opened']"
    groupedBy "Cover"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['MPC_open_MPC_isOpen']"
    isAffectedBy "['MPC_close_MPC_open']"
  ]
  node [
    id 48
    label "M_ErrorActive"
    type "['Variable']"
    iri "cg_store#M_ErrorActive"
    comment "['True = at least one error is active']"
    groupedBy "Monitoring"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['ErrorActive_to_WarnWithStacklight']"
    isAffectedBy "['HP_A_700304_to_ErrorActive', 'LP_A_700301_to_ErrorActive', 'CLT_A_700310_to_ErrorActive', 'Hyd_A_700202_to_ErrorActive', 'Hyd_A_700207_to_ErrorActive', 'Hyd_A_700206_to_ErrorActive', 'Hyd_A_700208_to_ErrorActive', 'Hyd_A_700204_to_ErrorActive']"
  ]
  node [
    id 49
    label "M_Lifebit"
    type "['Variable']"
    iri "cg_store#M_Lifebit"
    comment "['heartbeat Bit that alternates between True and False to indicate that communication to upper systems is healthy']"
    groupedBy "Monitoring"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
  ]
  node [
    id 50
    label "M_PowerOnDuration"
    type "['Variable']"
    iri "cg_store#M_PowerOnDuration"
    comment "['time in minutes since last normal power-up']"
    groupedBy "Monitoring"
    io "output"
    datatype "int"
    variabletype "counter"
    examples "93, 94, 95"
  ]
  node [
    id 51
    label "M_WarnActive"
    type "['Variable']"
    iri "cg_store#M_WarnActive"
    comment "['True = at least one warning is active, that does not trigger the stacklight']"
    groupedBy "Monitoring"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['WarnActive_to_SL_Yellow']"
    isAffectedBy "['Hyd_A_700205_to_WarnActive', 'Hyd_A_700203_to_WarnActive']"
  ]
  node [
    id 52
    label "M_WarnWithStacklight"
    type "['Variable']"
    iri "cg_store#M_WarnWithStacklight"
    comment "['True = at least one warning is active that triggers the stacklight']"
    groupedBy "Monitoring"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['WarnWithStacklight_to_SL_Red', 'WarnWithStacklight_to_SL_Yellow', 'WarnWithStacklight_to_SL_Green']"
    isAffectedBy "['ErrorActive_to_WarnWithStacklight', 'CLF_A_700307_to_WarnWithStacklight', 'MPA_A_701124_to_WarnWithStacklight', 'MPA_A_701125_to_WarnWithStacklight']"
  ]
  node [
    id 53
    label "SL_Green"
    type "['Variable']"
    iri "cg_store#SL_Green"
    comment "['True = stacklight indicates green (all good)']"
    groupedBy "Stacklight"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['WarnWithStacklight_to_SL_Green']"
  ]
  node [
    id 54
    label "SL_Red"
    type "['Variable']"
    iri "cg_store#SL_Red"
    comment "['True = stacklight indicates red (fault / stop)']"
    groupedBy "Stacklight"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['WarnWithStacklight_to_SL_Red']"
  ]
  node [
    id 55
    label "SL_Yellow"
    type "['Variable']"
    iri "cg_store#SL_Yellow"
    comment "['True = stacklight indicates yellow (notifications / warnings are present)']"
    groupedBy "Stacklight"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['WarnWithStacklight_to_SL_Yellow', 'WarnActive_to_SL_Yellow']"
  ]
  node [
    id 57
    label "SR_loosen"
    type "['Variable']"
    iri "cg_store#SR_loosen"
    comment "['True = steady rest commanded to loosen workpiece clamping']"
    groupedBy "Clamping"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['SR_loosen_SR_tighten']"
  ]
  node [
    id 58
    label "SR_tighten"
    type "['Variable']"
    iri "cg_store#SR_tighten"
    comment "['True = steady rest commanded to tighten workpiece clamping']"
    groupedBy "Clamping"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['SR_loosen_SR_tighten']"
  ]
  node [
    id 59
    label "SRL_lock"
    type "['Variable']"
    iri "cg_store#SRL_lock"
    comment "['True = steady rest is commanded to lock in position']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['SRL_lock_SRL_Locked']"
    isAffectedBy "['Prog_Name_Line_No_to_SRL_lock']"
  ]
  node [
    id 60
    label "SRL_Locked"
    type "['Variable']"
    iri "cg_store#SRL_Locked"
    comment "['True = steady rest is Locked in position']"
    groupedBy "Lock"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['SRL_lock_SRL_Locked']"
  ]
  node [
    id 61
    label "SRL_unlock"
    type "['Variable']"
    iri "cg_store#SRL_unlock"
    comment "['True = steady rest is commanded to unlock position']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['SRL_unlock_SRL_Unlocked']"
    isAffectedBy "['Prog_Name_Line_No_to_SRL_unlock']"
  ]
  node [
    id 62
    label "SRL_Unlocked"
    type "['Variable']"
    iri "cg_store#SRL_Unlocked"
    comment "['True = steady rest is Unlocked and free to move position']"
    groupedBy "Lock"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['SRL_unlock_SRL_Unlocked']"
  ]
  node [
    id 64
    label "Prog_CuttingTime"
    type "['Variable']"
    iri "cg_store#Prog_CuttingTime"
    comment "['Tool operating time (in seconds). The operating time of the path axes excluding active rapid traverse is measured in all NC programs between NC Start and Program End/NC Reset.\n                                                                The measurement is also interrupted during an active dwell time. The timer is automatically set to zero every time the control boots on default values.']"
    groupedBy "Program"
    io "output"
    datatype "float"
    variabletype "counter"
    examples "122155.05781798981, 122233.14181772165"
  ]
  node [
    id 65
    label "Prog_CycleTime"
    type "['Variable']"
    iri "cg_store#Prog_CycleTime"
    comment "['Runtime of the selected NC (sub-)program in seconds: The runtime between NC start and program end / NC reset is measured in the selected NC program. The timer is deleted when a new NC program is started.']"
    groupedBy "Program"
    io "output"
    datatype "float"
    variabletype "counter"
    examples "7.208, 8.208, 0.094, 1.284"
    isAffectedBy "['Prog_Name_Prog_CycleTime']"
  ]
  node [
    id 66
    label "Prog_LineNo"
    type "['Variable']"
    iri "cg_store#Prog_LineNo"
    comment "['Line number of the current NC block (from1) in current sub-program.\n                                                        0: before program start -1: not available due to error -2: not available due to DISPLOF']"
    groupedBy "Program"
    io "output"
    datatype "int"
    variabletype "categorical"
    examples "159, 173, 0, 125, 127"
    isCausing "['Prog_LineNo_Prog_Name_LineNo']"
    isAffectedBy "['Prog_Name_Prog_LineNo']"
  ]
  node [
    id 67
    label "Prog_Name"
    type "['Variable']"
    iri "cg_store#Prog_Name"
    comment "['Program name of the currently active (sub-)program']"
    groupedBy "Program"
    io "output"
    datatype "str"
    variabletype "categorical"
    examples "MAIN_CH1_MPF, HOME_CH1_SPF, MEAS_SEMI_FINISH_PT_SPF"
    isCausing "['Prog_Name_Prog_LineNo', 'Prog_Name_Prog_Name_LineNo', 'Prog_Name_Prog_CycleTime', 'Prog_Name_T_A_701309']"
  ]
  node [
    id 68
    label "Prog_Name_LineNo"
    type "['Variable']"
    iri "cg_store#Prog_Name_LineNo"
    comment "['Joined identifier of current active sub-program and line number within subprogram in pattern: <Prog_Name>|<LineNo>']"
    groupedBy "Program"
    io "output"
    datatype "str"
    variabletype "categorical"
    examples "MAIN_CH1_MPF|-2, HOME_CH1_SPF|10, MEAS_SEMI_FINISH_PT_SPF|159"
    isCausing "['Prog_Name_Prog_A_701330', 'Prog_Name_Line_No_to_SR_A_67040', 'Prog_Name_Line_No_to_SRL_unlock', 'Prog_Name_Line_No_to_SRL_lock', 'Prog_Name_Line_No_to_Spd_InnerCool', 'Prog_Name_Line_No_to_Spd_OuterCool', 'Prog_Name_Line_No_to_TL_unlock', 'Prog_Name_Line_No_to_TL_lock', 'Prog_Name_Line_No_to_CBC_open', 'Prog_Name_Line_No_to_CBC_close', 'Prog_Name_Line_No_to_WC_inPos', 'Prog_Name_LineNo_CBC_open', 'Prog_Name_LineNo_CBC_close', 'Prog_Name_LineNo_WCL_toInitPos', 'Prog_Name_LineNo_WCL_toWorkPos']"
    isAffectedBy "['Prog_Name_Prog_Name_LineNo', 'Prog_LineNo_Prog_Name_LineNo']"
  ]
  node [
    id 69
    label "Spd_InnerCoolOn"
    type "['Variable']"
    iri "cg_store#Spd_InnerCoolOn"
    comment "['True = inner cooling on spindle is commanded to turn on']"
    groupedBy "Spindle"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['Spd_InnerCoolOn_LP_Pump_On']"
    isAffectedBy "['Prog_Name_Line_No_to_Spd_InnerCool']"
  ]
  node [
    id 70
    label "Spd_OuterCoolOn"
    type "['Variable']"
    iri "cg_store#Spd_OuterCoolOn"
    comment "['True = outer cooling on spindle is commanded to turn on']"
    groupedBy "Spindle"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['Prog_Name_Line_No_to_Spd_OuterCool']"
  ]
  node [
    id 71
    label "Spd_ActPos_C"
    type "['Variable']"
    iri "cg_store#Spd_ActPos_C"
    comment "['actual position of the spindles spinning c-axis in degrees']"
    groupedBy "cAxis"
    io "input"
    datatype "float"
    variabletype "continuous"
    examples "4.142, 358.075"
    isAffectedBy "['Spd_ActSpeed_C_Spd_ActPos_C']"
  ]
  node [
    id 72
    label "Spd_ActSpeed_C"
    type "['Variable']"
    iri "cg_store#Spd_ActSpeed_C"
    comment "['actual speed of the spindles spinning c-axis in degrees/second']"
    groupedBy "cAxis"
    io "input"
    datatype "float"
    variabletype "continuous"
    examples "0.0, -30000.0, 16103.6"
    isCausing "['Spd_ActSpeed_C_Spd_ActPos_C']"
  ]
  node [
    id 73
    label "Spd_ActPos_X"
    type "['Variable']"
    iri "cg_store#Spd_ActPos_X"
    comment "['actual position of the spindles x-axis in mm']"
    groupedBy "xAxis"
    io "input"
    datatype "float"
    variabletype "continuous"
    examples "382.977, 612.124, 60.23"
    isAffectedBy "['Spd_ActSpeed_X_Spd_ActPos_X']"
  ]
  node [
    id 74
    label "Spd_ActSpeed_X"
    type "['Variable']"
    iri "cg_store#Spd_ActSpeed_X"
    comment "['actual speed of the spindles x-axis in mm/s']"
    groupedBy "xAxis"
    io "input"
    datatype "float"
    variabletype "continuous"
    examples "0.0, 34320.0, -3480.0"
    isCausing "['Spd_ActSpeed_X_Spd_ActPos_X']"
  ]
  node [
    id 75
    label "Spd_ActPos_Z"
    type "['Variable']"
    iri "cg_store#Spd_ActPos_Z"
    comment "['actual position of the spindles z-axis in mm']"
    groupedBy "zAxis"
    io "input"
    datatype "float"
    variabletype "continuous"
    examples "250.325, 495.0"
    isAffectedBy "['Spd_ActSpeed_Z_Spd_ActPos_Z']"
  ]
  node [
    id 76
    label "Spd_ActSpeed_Z"
    type "['Variable']"
    iri "cg_store#Spd_ActSpeed_Z"
    comment "['actual speed of the spindles z-axis in mm/s']"
    groupedBy "zAxis"
    io "input"
    datatype "float"
    variabletype "continuous"
    examples "0.0, 34320.0, -3480.0"
    isCausing "['Spd_ActSpeed_Z_Spd_ActPos_Z']"
  ]
  node [
    id 78
    label "T_ActToolIdent"
    type "['Variable']"
    iri "cg_store#T_ActToolIdent"
    comment "['string identifier of currently clamped tool']"
    groupedBy "Tool"
    io "input"
    datatype "str"
    variabletype "categorical"
    examples "3D_PROBE', '002398_A'"
    isCausing "['T_ActToolIdent_TL_lock']"
    isAffectedBy "['TL_unlock_T_ActToolIdent']"
  ]
  node [
    id 79
    label "TL_lock"
    type "['Variable']"
    iri "cg_store#TL_lock"
    comment "['True = tool revolver is commanded to lock in position']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['TL_lock_TL_Locked']"
    isAffectedBy "['Prog_Name_Line_No_to_TL_lock', 'TL_unlock_TL_lock', 'T_ActToolIdent_TL_lock']"
  ]
  node [
    id 80
    label "TL_Locked"
    type "['Variable']"
    iri "cg_store#TL_Locked"
    comment "['True = tool revolver is Locked in position']"
    groupedBy "Lock"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['TL_lock_TL_Locked', 'TL_unlock_TL_Locked']"
  ]
  node [
    id 81
    label "TL_unlock"
    type "['Variable']"
    iri "cg_store#TL_unlock"
    comment "['True = tool revolver commanded to unlock position (required for tool change)']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['TL_unlock_TL_Locked', 'TL_unlock_TL_lock', 'TL_unlock_T_ActToolIdent']"
    isAffectedBy "['Prog_Name_Line_No_to_TL_unlock', 'T_A_701309_TL_unlock']"
  ]
  node [
    id 82
    label "CBC_close"
    type "['Variable']"
    iri "cg_store#CBC_close"
    comment "['True = cover protecting conveyer belt with workpiece carriers in workspace is commanded to be closed']"
    groupedBy "ConveyerBeltCover"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['CBC_close_CBC_Closed']"
    isAffectedBy "['Prog_Name_Line_No_to_CBC_close', 'WCL_InitPos_CBC_close', 'Prog_Name_LineNo_CBC_close']"
  ]
  node [
    id 83
    label "CBC_Closed"
    type "['Variable']"
    iri "cg_store#CBC_Closed"
    comment "['True = cover protecting conveyer belt with workpiece carriers in workspace is closed']"
    groupedBy "ConveyerBeltCover"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['CBC_close_CBC_Closed']"
  ]
  node [
    id 84
    label "CBC_isOpen"
    type "['Variable']"
    iri "cg_store#CBC_isOpen"
    comment "['True = cover protecting conveyer belt with workpiece carriers in workspace is open']"
    groupedBy "ConveyerBeltCover"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['CBC_isOpen_WCL_toWorkPos']"
    isAffectedBy "['CBC_open_CBC_isOpen']"
  ]
  node [
    id 85
    label "CBC_open"
    type "['Variable']"
    iri "cg_store#CBC_open"
    comment "['True = cover protecting conveyer belt with workpiece carriers in workspace to be opened']"
    groupedBy "ConveyerBeltCover"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['CBC_open_CBC_isOpen']"
    isAffectedBy "['Prog_Name_Line_No_to_CBC_open', 'Prog_Name_LineNo_CBC_open']"
  ]
  node [
    id 86
    label "WC_inPos"
    type "['Variable']"
    iri "cg_store#WC_inPos"
    comment "['True = sensor indicates that workpiece carrier is in correct position']"
    groupedBy "WorkpieceCarrier"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['Prog_Name_Line_No_to_WC_inPos']"
  ]
  node [
    id 87
    label "WCL_InitPos"
    type "['Variable']"
    iri "cg_store#WCL_InitPos"
    comment "['True = lock for securing workpiece carrier is in initial position = hidden']"
    groupedBy "Lock"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['WCL_InitPos_CBC_close']"
    isAffectedBy "['WCL_toInitPos_WCL_InitPos']"
  ]
  node [
    id 88
    label "WCL_Locked"
    type "['Variable']"
    iri "cg_store#WCL_Locked"
    comment "['True = workpiece carrier is locked']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isAffectedBy "['WCL_WorkPos_WCL_Locked']"
  ]
  node [
    id 89
    label "WCL_toInitPos"
    type "['Variable']"
    iri "cg_store#WCL_toInitPos"
    comment "['True = lock for securing workpiece carrier commanded to initial position (hidden)']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['WCL_toInitPos_WCL_InitPos']"
    isAffectedBy "['WCL_Unlocked_WCL_toInitPos', 'Prog_Name_LineNo_WCL_toInitPos']"
  ]
  node [
    id 90
    label "WCL_toWorkPos"
    type "['Variable']"
    iri "cg_store#WCL_toWorkPos"
    comment "['True = lock for securing workpiece carrier commanded to working position, where it can lock the workpiece carrier']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['WCL_toWorkPos_WCL_WorkPos']"
    isAffectedBy "['CBC_isOpen_WCL_toWorkPos', 'Prog_Name_LineNo_WCL_toWorkPos']"
  ]
  node [
    id 91
    label "WCL_Unlocked"
    type "['Variable']"
    iri "cg_store#WCL_Unlocked"
    comment "['True = workpiece carrier is unlocked']"
    groupedBy "Lock"
    io "output"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['WCL_Unlocked_WCL_toInitPos']"
  ]
  node [
    id 92
    label "WCL_WorkPos"
    type "['Variable']"
    iri "cg_store#WCL_WorkPos"
    comment "['True = lock for securing workpiece carrier is in working pos, where it can lock the workpiece carrier']"
    groupedBy "Lock"
    io "input"
    datatype "bool"
    variabletype "binary"
    examples "True, False"
    isCausing "['WCL_WorkPos_WCL_Locked']"
    isAffectedBy "['WCL_toWorkPos_WCL_WorkPos']"
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
    isCausing "['CLF_A_700307_to_WarnWithStacklight']"
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
    isCausing "['HP_A_700304_to_ErrorActive']"
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
    isCausing "['LP_A_700301_to_ErrorActive']"
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
    isCausing "['CLT_A_700310_to_ErrorActive']"
    isAffectedBy "['CLT_Level_Min_Monitoring']"
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
    isCausing "['Hyd_A_700202_Disables_Hyd_Valve_P_Up', 'Hyd_A_700202_Disables_Hyd_Pump_On', 'Hyd_A_700202_to_ErrorActive']"
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
    isCausing "['Hyd_A_700207_Disables_Hyd_IsEnabled', 'Hyd_A_700207_to_ErrorActive']"
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
    isCausing "['Hyd_A_700205_to_WarnActive']"
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
    isCausing "['Hyd_A_700206_Disables_Hyd_Pump_On', 'Hyd_A_700206_to_ErrorActive']"
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
    isCausing "['Hyd_A_700208_to_ErrorActive']"
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
    isCausing "['Hyd_A_700203_to_WarnActive']"
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
    isCausing "['Hyd_A_700204_Disables_Hyd_IsEnabled', 'Hyd_A_700204_to_ErrorActive']"
    isAffectedBy "['Hyd_Temp_Alarm_Monitoring']"
  ]
  node [
    id 38
    label "MPA_A_701124"
    type "['Alarm']"
    iri "cg_store#MPA_A_701124"
    comment "['alarm that for measuring probe did not reach end position in time']"
    groupedBy "Axis"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isCausing "['MPA_A_701124_to_WarnWithStacklight']"
    isAffectedBy "['MPA_WorkPos_Monitoring', 'MPA_toWorkPos_Monitoring', 'MPA_InitPos_Monitoring', 'MPA_toInitPos_Monitoring']"
  ]
  node [
    id 39
    label "MPA_A_701125"
    type "['Alarm']"
    iri "cg_store#MPA_A_701125"
    comment "['alarm that for measuring probe axis both endposition sensor inputs indicate true']"
    groupedBy "Axis"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isCausing "['MPA_A_701125_to_WarnWithStacklight']"
    isAffectedBy "['MPA_WorkPos_BothEndPosSensors', 'MPA_InitPos_BothEndPosSensors']"
  ]
  node [
    id 56
    label "SR_A_67040"
    type "['Alarm']"
    iri "cg_store#SR_A_67040"
    comment "['notification that moving steady rest is already in home position']"
    groupedBy "MovingSteadyRest"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['Prog_Name_Line_No_to_SR_A_67040']"
  ]
  node [
    id 63
    label "Prog_A_701330"
    type "['Alarm']"
    iri "cg_store#Prog_A_701330"
    comment "['notification that home position is reached - continue with NC start']"
    groupedBy "Program"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isAffectedBy "['Prog_Name_Prog_A_701330']"
  ]
  node [
    id 77
    label "T_A_701309"
    type "['Alarm']"
    iri "cg_store#T_A_701309"
    comment "['warning that tool change is in progress']"
    groupedBy "Tool"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
    isCausing "['T_A_701309_TL_unlock']"
    isAffectedBy "['Prog_Name_T_A_701309']"
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
  edge [
    source 17
    target 16
    key 0
    type "['CausalEdge']"
    iri "cg_store#ExU_On_ExU_isOff"
    comment "['When ExtractionUnit is commanded to be on (ExU_On=True), it results in the ExtractionUnit being on (ExU_isOff=False).']"
    hasCause "ExU_On"
    hasEffect "ExU_isOff"
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
    source 35
    target 36
    key 0
    type "['CausalEdge']"
    iri "cg_store#Lubr_On_Lubr_P_Ok"
    comment "['When lubrication is commanded On (Lubr_On=True), pressure builds up until sufficient (Lubr_P_Ok=True).\n                                When LubrOn=False, Lubr_P_Ok also changes to False.']"
    hasCause "Lubr_On"
    hasEffect "Lubr_P_Ok"
  ]
  edge [
    source 40
    target 38
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_InitPos_Monitoring"
    comment "['When axis is not reaching initial pos in time (MPA_InitPos=False), besides being commanded to (MPA_toInitPos=True),\n                        end position not reached alarm 701124 is triggered.']"
    hasCause "MPA_InitPos"
    hasEffect "MPA_A_701124"
  ]
  edge [
    source 40
    target 39
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_InitPos_BothEndPosSensors"
    comment "['When both end position sensors are triggered (MPA_WorkPos=True AND MPA_InitPos=True),\n                        alarm 701125 is triggered.']"
    hasCause "MPA_InitPos"
    hasEffect "MPA_A_701125"
  ]
  edge [
    source 40
    target 44
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_InitPos_MPC_close"
    comment "['When axis is in initial position (MPA_InitPos=True), cover can be commanded to close (MPC_close=True)']"
    hasCause "MPA_InitPos"
    hasEffect "MPC_close"
  ]
  edge [
    source 41
    target 40
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_toInitPos_MPA_InitPos"
    comment "['When axis is commanded to initial position (MPA_toInitPos=True), it moves to initial position (MPA_InitPos=True)']"
    hasCause "MPA_toInitPos"
    hasEffect "MPA_InitPos"
  ]
  edge [
    source 41
    target 42
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_toInitPos_MPA_toWorkPos"
    comment "['When axis is not commanded to initial position (MPA_toInitPos=False),\n                                it can be commanded to working position (MPA_toWorkPos=True)']"
    hasCause "MPA_toInitPos"
    hasEffect "MPA_toWorkPos"
  ]
  edge [
    source 41
    target 38
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_toInitPos_Monitoring"
    comment "['When axis is commanded to initial position (MPA_toInitPos=True) but does not reach it in time (MPA_InitPos=False),\n                        end position not reached alarm 701124 is triggered.']"
    hasCause "MPA_toInitPos"
    hasEffect "MPA_A_701124"
  ]
  edge [
    source 42
    target 43
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_toWorkPos_MPA_WorkPos"
    comment "['When axis is commanded to working position (MPA_toWorkPos=True), it moves to working position (MPA_WorkPos=True)']"
    hasCause "MPA_toWorkPos"
    hasEffect "MPA_WorkPos"
  ]
  edge [
    source 42
    target 38
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_toWorkPos_Monitoring"
    comment "['When axis is commanded to working position (MPA_toWorkPos=True) but does not reach it in time (MPA_WorkPos=False),\n                        end position not reached alarm 701124 is triggered.']"
    hasCause "MPA_toWorkPos"
    hasEffect "MPA_A_701124"
  ]
  edge [
    source 43
    target 37
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_WorkPos_MP_Inactive"
    comment "['When axis is in working position (MPA_WorkPos=True), measuring probe can be activated (MP_Inactive=False)']"
    hasCause "MPA_WorkPos"
    hasEffect "MP_Inactive"
  ]
  edge [
    source 43
    target 38
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_WorkPos_Monitoring"
    comment "['When axis is not reaching work pos in time (MPA_WorkPos=False), besides being commanded to (MPA_toWorkPos=True),\n                        end position not reached alarm 701124 is triggered.']"
    hasCause "MPA_WorkPos"
    hasEffect "MPA_A_701124"
  ]
  edge [
    source 43
    target 39
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_WorkPos_BothEndPosSensors"
    comment "['When both end position sensors are triggered (MPA_WorkPos=True AND MPA_InitPos=True),\n                        alarm 701125 is triggered.']"
    hasCause "MPA_WorkPos"
    hasEffect "MPA_A_701125"
  ]
  edge [
    source 44
    target 45
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPC_close_MPC_Closed"
    comment "['When cover is commanded to close (MPC_close=True), cover closes (MPC_Closed=True)']"
    hasCause "MPC_close"
    hasEffect "MPC_Closed"
  ]
  edge [
    source 44
    target 47
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPC_close_MPC_open"
    comment "['When cover is not commanded to close (MPC_close=False),\n                                it can be commanded to open (MPC_open=True)']"
    hasCause "MPC_close"
    hasEffect "MPC_open"
  ]
  edge [
    source 46
    target 42
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPC_isOpen_MPA_toWorkPos"
    comment "['When cover is open (MPC_isOpen=True), axis can be commanded to working position (MPA_toWorkPos=True)']"
    hasCause "MPC_isOpen"
    hasEffect "MPA_toWorkPos"
  ]
  edge [
    source 47
    target 46
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPC_open_MPC_isOpen"
    comment "['When cover is commanded to open (MPC_open=True), cover opens (MPC_isOpen=True)']"
    hasCause "MPC_open"
    hasEffect "MPC_isOpen"
  ]
  edge [
    source 48
    target 52
    key 0
    type "['CausalEdge']"
    iri "cg_store#ErrorActive_to_WarnWithStacklight"
    comment "['When an error is active, a warning with stacklight is also triggered.']"
    hasCause "M_ErrorActive"
    hasEffect "M_WarnWithStacklight"
  ]
  edge [
    source 51
    target 55
    key 0
    type "['CausalEdge']"
    iri "cg_store#WarnActive_to_SL_Yellow"
    comment "['A general warning can affect the yellow light state (e.g., blinking).']"
    hasCause "M_WarnActive"
    hasEffect "SL_Yellow"
  ]
  edge [
    source 52
    target 54
    key 0
    type "['CausalEdge']"
    iri "cg_store#WarnWithStacklight_to_SL_Red"
    comment "['A warning with stacklight can affect the red light state (e.g., blinking).']"
    hasCause "M_WarnWithStacklight"
    hasEffect "SL_Red"
  ]
  edge [
    source 52
    target 55
    key 0
    type "['CausalEdge']"
    iri "cg_store#WarnWithStacklight_to_SL_Yellow"
    comment "['A warning with stacklight can affect the yellow light state (e.g., blinking).']"
    hasCause "M_WarnWithStacklight"
    hasEffect "SL_Yellow"
  ]
  edge [
    source 52
    target 53
    key 0
    type "['CausalEdge']"
    iri "cg_store#WarnWithStacklight_to_SL_Green"
    comment "['A warning with stacklight can affect the green light state (e.g., blinking).']"
    hasCause "M_WarnWithStacklight"
    hasEffect "SL_Green"
  ]
  edge [
    source 57
    target 58
    key 0
    type "['CausalEdge']"
    iri "cg_store#SR_loosen_SR_tighten"
    comment "['When steady rest is commanded to loosen (SR_loosen=False), it can be commanded to tighten (SR_tighten=True)']"
    hasCause "SR_loosen"
    hasEffect "SR_tighten"
  ]
  edge [
    source 59
    target 60
    key 0
    type "['CausalEdge']"
    iri "cg_store#SRL_lock_SRL_Locked"
    comment "['When steady rest is commanded to lock (SRL_lock=True), it results in the steady rest being locked (SRL_Locked=True)']"
    hasCause "SRL_lock"
    hasEffect "SRL_Locked"
  ]
  edge [
    source 61
    target 62
    key 0
    type "['CausalEdge']"
    iri "cg_store#SRL_unlock_SRL_Unlocked"
    comment "['When steady rest is commanded to unlock (SRL_unlock=True), it results in the steady rest being unlocked (SRL_Unlocked=True)']"
    hasCause "SRL_unlock"
    hasEffect "SRL_Unlocked"
  ]
  edge [
    source 66
    target 68
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_LineNo_Prog_Name_LineNo"
    comment "['The current line number (Prog_LineNo) is used to construct the joined identifier (Prog_Name_LineNo)']"
    hasCause "Prog_LineNo"
    hasEffect "Prog_Name_LineNo"
  ]
  edge [
    source 67
    target 66
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Prog_LineNo"
    comment "['The active program (Prog_Name) determines which line numbers (Prog_LineNo) are available']"
    hasCause "Prog_Name"
    hasEffect "Prog_LineNo"
  ]
  edge [
    source 67
    target 68
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Prog_Name_LineNo"
    comment "['The active program name (Prog_Name) is used to construct the joined identifier (Prog_Name_LineNo)']"
    hasCause "Prog_Name"
    hasEffect "Prog_Name_LineNo"
  ]
  edge [
    source 67
    target 65
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Prog_CycleTime"
    comment "['The cycle time (Prog_CycleTime) is measured for the currently active program (Prog_Name)']"
    hasCause "Prog_Name"
    hasEffect "Prog_CycleTime"
  ]
  edge [
    source 67
    target 77
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_T_A_701309"
    comment "['Certain programs trigger the tool change alarm (T_A_701309=True)']"
    hasCause "Prog_Name"
    hasEffect "T_A_701309"
  ]
  edge [
    source 68
    target 63
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Prog_A_701330"
    comment "['Certain programs like HOME_CH1_SPF and specific lines may trigger the home position notification (Prog_A_701330)']"
    hasCause "Prog_Name_LineNo"
    hasEffect "Prog_A_701330"
  ]
  edge [
    source 68
    target 56
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_SR_A_67040"
    comment "['Prog_name HOME_CH1_SPF around line 127 wants to home moving steady rest, if already in home pos: notification']"
    hasCause "Prog_Name_LineNo"
    hasEffect "SR_A_67040"
  ]
  edge [
    source 68
    target 61
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_SRL_unlock"
    comment "['Certain program line numbers start tool loading / unloading']"
    hasCause "Prog_Name_LineNo"
    hasEffect "SRL_unlock"
  ]
  edge [
    source 68
    target 59
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_SRL_lock"
    comment "['Certain program line numbers start tool loading / unloading']"
    hasCause "Prog_Name_LineNo"
    hasEffect "SRL_lock"
  ]
  edge [
    source 68
    target 69
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_Spd_InnerCool"
    comment "['Machining program controls cooling.']"
    hasCause "Prog_Name_LineNo"
    hasEffect "Spd_InnerCoolOn"
  ]
  edge [
    source 68
    target 70
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_Spd_OuterCool"
    comment "['Machining program controls cooling.']"
    hasCause "Prog_Name_LineNo"
    hasEffect "Spd_OuterCoolOn"
  ]
  edge [
    source 68
    target 81
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_TL_unlock"
    comment "['Certain program line numbers trigger tool change operations']"
    hasCause "Prog_Name_LineNo"
    hasEffect "TL_unlock"
  ]
  edge [
    source 68
    target 79
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_TL_lock"
    comment "['Certain program line numbers complete tool change operations']"
    hasCause "Prog_Name_LineNo"
    hasEffect "TL_lock"
  ]
  edge [
    source 68
    target 85
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_CBC_open"
    comment "['Certain program line numbers command the cover to open (CBC_open=True)']"
    hasCause "Prog_Name_LineNo"
    hasEffect "CBC_open"
  ]
  edge [
    source 68
    target 85
    key 1
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_LineNo_CBC_open"
    comment "['Certain program line numbers trigger cover opening']"
    hasCause "Prog_Name_LineNo"
    hasEffect "CBC_open"
  ]
  edge [
    source 68
    target 82
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_CBC_close"
    comment "['Certain program line numbers command the cover to close (CBC_close=True)']"
    hasCause "Prog_Name_LineNo"
    hasEffect "CBC_close"
  ]
  edge [
    source 68
    target 82
    key 1
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_LineNo_CBC_close"
    comment "['Certain program line numbers trigger cover closing']"
    hasCause "Prog_Name_LineNo"
    hasEffect "CBC_close"
  ]
  edge [
    source 68
    target 86
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_Line_No_to_WC_inPos"
    comment "['Certain program line numbers check if workpiece carrier is in position (WC_inPos=True)']"
    hasCause "Prog_Name_LineNo"
    hasEffect "WC_inPos"
  ]
  edge [
    source 68
    target 89
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_LineNo_WCL_toInitPos"
    comment "['Certain program line numbers trigger lock to move to initial position']"
    hasCause "Prog_Name_LineNo"
    hasEffect "WCL_toInitPos"
  ]
  edge [
    source 68
    target 90
    key 0
    type "['CausalEdge']"
    iri "cg_store#Prog_Name_LineNo_WCL_toWorkPos"
    comment "['Certain program line numbers trigger lock to move to working position']"
    hasCause "Prog_Name_LineNo"
    hasEffect "WCL_toWorkPos"
  ]
  edge [
    source 69
    target 13
    key 0
    type "['CausalEdge']"
    iri "cg_store#Spd_InnerCoolOn_LP_Pump_On"
    comment "['When inner spindle coolant is on, pump needs to generate more pressure and is turned on.']"
    hasCause "Spd_InnerCoolOn"
    hasEffect "LP_Pump_On"
  ]
  edge [
    source 72
    target 71
    key 0
    type "['CausalEdge']"
    iri "cg_store#Spd_ActSpeed_C_Spd_ActPos_C"
    comment "['Speed of the spindle c-axis (Spd_ActSpeed_C) affects its position (Spd_ActPos_C) over time.']"
    hasCause "Spd_ActSpeed_C"
    hasEffect "Spd_ActPos_C"
  ]
  edge [
    source 74
    target 73
    key 0
    type "['CausalEdge']"
    iri "cg_store#Spd_ActSpeed_X_Spd_ActPos_X"
    comment "['Speed of the spindle x-axis (Spd_ActSpeed_X) affects its position (Spd_ActPos_X) over time.']"
    hasCause "Spd_ActSpeed_X"
    hasEffect "Spd_ActPos_X"
  ]
  edge [
    source 76
    target 75
    key 0
    type "['CausalEdge']"
    iri "cg_store#Spd_ActSpeed_Z_Spd_ActPos_Z"
    comment "['Speed of the spindle z-axis (Spd_ActSpeed_Z) affects its position over time (Spd_ActPos_Z).']"
    hasCause "Spd_ActSpeed_Z"
    hasEffect "Spd_ActPos_Z"
  ]
  edge [
    source 78
    target 79
    key 0
    type "['CausalEdge']"
    iri "cg_store#T_ActToolIdent_TL_lock"
    comment "['When a new tool is detected (T_ActToolIdent changes), the tool revolver is commanded to lock (TL_lock=True)']"
    hasCause "T_ActToolIdent"
    hasEffect "TL_lock"
  ]
  edge [
    source 79
    target 80
    key 0
    type "['CausalEdge']"
    iri "cg_store#TL_lock_TL_Locked"
    comment "['When tool revolver is commanded to lock (TL_lock=True), it results in the revolver being locked (TL_Locked=True)']"
    hasCause "TL_lock"
    hasEffect "TL_Locked"
  ]
  edge [
    source 81
    target 80
    key 0
    type "['CausalEdge']"
    iri "cg_store#TL_unlock_TL_Locked"
    comment "['When tool revolver is commanded to unlock (TL_unlock=True), it results in the revolver being unlocked (TL_Locked=False)']"
    hasCause "TL_unlock"
    hasEffect "TL_Locked"
  ]
  edge [
    source 81
    target 79
    key 0
    type "['CausalEdge']"
    iri "cg_store#TL_unlock_TL_lock"
    comment "['When tool revolver is commanded to unlock (TL_unlock=True), it prevents the lock command (TL_lock=False)']"
    hasCause "TL_unlock"
    hasEffect "TL_lock"
  ]
  edge [
    source 81
    target 78
    key 0
    type "['CausalEdge']"
    iri "cg_store#TL_unlock_T_ActToolIdent"
    comment "['When tool revolver is unlocked (TL_unlock=True), it allows for tool change, which updates the active tool identifier']"
    hasCause "TL_unlock"
    hasEffect "T_ActToolIdent"
  ]
  edge [
    source 82
    target 83
    key 0
    type "['CausalEdge']"
    iri "cg_store#CBC_close_CBC_Closed"
    comment "['When cover is commanded to close (CBC_close=True), it results in the cover being closed (CBC_Closed=True)']"
    hasCause "CBC_close"
    hasEffect "CBC_Closed"
  ]
  edge [
    source 84
    target 90
    key 0
    type "['CausalEdge']"
    iri "cg_store#CBC_isOpen_WCL_toWorkPos"
    comment "['When cover is open (CBC_isOpen=True), workpiece carrier lock can be moved into position']"
    hasCause "CBC_isOpen"
    hasEffect "WCL_toWorkPos"
  ]
  edge [
    source 85
    target 84
    key 0
    type "['CausalEdge']"
    iri "cg_store#CBC_open_CBC_isOpen"
    comment "['When cover is commanded to open (CBC_open=True), it results in the cover being open (CBC_isOpen=True)']"
    hasCause "CBC_open"
    hasEffect "CBC_isOpen"
  ]
  edge [
    source 87
    target 82
    key 0
    type "['CausalEdge']"
    iri "cg_store#WCL_InitPos_CBC_close"
    comment "['When workpiece carrier lock is back in initial position (WCL_InitPos=True), the cover can be commanded to close (CBC_close=True)']"
    hasCause "WCL_InitPos"
    hasEffect "CBC_close"
  ]
  edge [
    source 89
    target 87
    key 0
    type "['CausalEdge']"
    iri "cg_store#WCL_toInitPos_WCL_InitPos"
    comment "['When lock is commanded to initial position (WCL_toInitPos=True), it moves to initial position (WCL_InitPos=True)']"
    hasCause "WCL_toInitPos"
    hasEffect "WCL_InitPos"
  ]
  edge [
    source 90
    target 92
    key 0
    type "['CausalEdge']"
    iri "cg_store#WCL_toWorkPos_WCL_WorkPos"
    comment "['When lock is commanded to working position (WCL_toWorkPos=True), it moves to working position (WCL_WorkPos=True)']"
    hasCause "WCL_toWorkPos"
    hasEffect "WCL_WorkPos"
  ]
  edge [
    source 91
    target 89
    key 0
    type "['CausalEdge']"
    iri "cg_store#WCL_Unlocked_WCL_toInitPos"
    comment "['Only when unlocked (WCL_Unlocked=True), lock can be commanded to initial position (WCL_toInitPos=True)']"
    hasCause "WCL_Unlocked"
    hasEffect "WCL_toInitPos"
  ]
  edge [
    source 92
    target 88
    key 0
    type "['CausalEdge']"
    iri "cg_store#WCL_WorkPos_WCL_Locked"
    comment "['When lock is in working position (WCL_WorkPos=True), carrier can be locked (WCL_Locked=True)']"
    hasCause "WCL_WorkPos"
    hasEffect "WCL_Locked"
  ]
  edge [
    source 1
    target 52
    key 0
    type "['CausalEdge']"
    iri "cg_store#CLF_A_700307_to_WarnWithStacklight"
    comment "['Clogged coolant filter triggers a warning with stacklight.']"
    hasCause "CLF_A_700307"
    hasEffect "M_WarnWithStacklight"
  ]
  edge [
    source 5
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#HP_A_700304_to_ErrorActive"
    comment "['High pressure pump motor protection alarm triggers an error state.']"
    hasCause "HP_A_700304"
    hasEffect "M_ErrorActive"
  ]
  edge [
    source 11
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#LP_A_700301_to_ErrorActive"
    comment "['Low pressure pump motor protection alarm triggers an error state.']"
    hasCause "LP_A_700301"
    hasEffect "M_ErrorActive"
  ]
  edge [
    source 14
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#CLT_A_700310_to_ErrorActive"
    comment "['Coolant tank level minimum alarm triggers an error state.']"
    hasCause "CLT_A_700310"
    hasEffect "M_ErrorActive"
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
    source 18
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700202_to_ErrorActive"
    comment "['Hydraulic pressure out of norm alarm triggers an error state.']"
    hasCause "Hyd_A_700202"
    hasEffect "M_ErrorActive"
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
    source 22
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700207_to_ErrorActive"
    comment "['Clogged hydraulic filter alarm triggers an error state.']"
    hasCause "Hyd_A_700207"
    hasEffect "M_ErrorActive"
  ]
  edge [
    source 24
    target 51
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700205_to_WarnActive"
    comment "['Hydraulic oil level near minimum warning triggers a general warning state.']"
    hasCause "Hyd_A_700205"
    hasEffect "M_WarnActive"
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
    source 25
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700206_to_ErrorActive"
    comment "['Hydraulic oil level below minimum alarm triggers an error state.']"
    hasCause "Hyd_A_700206"
    hasEffect "M_ErrorActive"
  ]
  edge [
    source 27
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700208_to_ErrorActive"
    comment "['Hydraulic pump motor protection alarm triggers an error state.']"
    hasCause "Hyd_A_700208"
    hasEffect "M_ErrorActive"
  ]
  edge [
    source 31
    target 51
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700203_to_WarnActive"
    comment "['High hydraulic temperature warning triggers a general warning state.']"
    hasCause "Hyd_A_700203"
    hasEffect "M_WarnActive"
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
  edge [
    source 32
    target 48
    key 0
    type "['CausalEdge']"
    iri "cg_store#Hyd_A_700204_to_ErrorActive"
    comment "['Critical hydraulic temperature alarm triggers an error state.']"
    hasCause "Hyd_A_700204"
    hasEffect "M_ErrorActive"
  ]
  edge [
    source 38
    target 52
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_A_701124_to_WarnWithStacklight"
    comment "['Measuring probe end position not reached alarm triggers a warning with stacklight.']"
    hasCause "MPA_A_701124"
    hasEffect "M_WarnWithStacklight"
  ]
  edge [
    source 39
    target 52
    key 0
    type "['CausalEdge']"
    iri "cg_store#MPA_A_701125_to_WarnWithStacklight"
    comment "['Measuring probe both end position sensors active alarm triggers a warning with stacklight.']"
    hasCause "MPA_A_701125"
    hasEffect "M_WarnWithStacklight"
  ]
  edge [
    source 77
    target 81
    key 0
    type "['CausalEdge']"
    iri "cg_store#T_A_701309_TL_unlock"
    comment "['When tool change alarm is active (T_A_701309=True), it triggers the tool revolver to unlock (TL_unlock=True)']"
    hasCause "T_A_701309"
    hasEffect "TL_unlock"
  ]
]
