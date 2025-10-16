graph [
  directed 1
  multigraph 1
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
    id 38
    label "MPA_A_701124"
    type "['Alarm']"
    iri "cg_store#MPA_A_701124"
    comment "['alarm that for measuring probe did not reach end position in time']"
    groupedBy "Axis"
    datatype "bool"
    variabletype "alarm"
    examples "True (=alarm active), False(=alarm inactive)"
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
    isAffectedBy "['MPA_WorkPos_BothEndPosSensors', 'MPA_InitPos_BothEndPosSensors']"
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
]
