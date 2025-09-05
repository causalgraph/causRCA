graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "MPA_A_701124"
  ]
  node [
    id 1
    label "MPA_A_701125"
  ]
  node [
    id 2
    label "MPA_InitPos"
  ]
  node [
    id 3
    label "MPA_WorkPos"
  ]
  node [
    id 4
    label "MPA_toInitPos"
  ]
  node [
    id 5
    label "MPA_toWorkPos"
  ]
  node [
    id 6
    label "MPC_Closed"
  ]
  node [
    id 7
    label "MPC_close"
  ]
  node [
    id 8
    label "MPC_isOpen"
  ]
  node [
    id 9
    label "MPC_open"
  ]
  node [
    id 10
    label "MP_Inactive"
  ]
  edge [
    source 1
    target 2
    key 0
    weight 1
  ]
  edge [
    source 1
    target 3
    key 0
    weight 1
  ]
  edge [
    source 2
    target 1
    key 0
    weight 1
  ]
  edge [
    source 2
    target 8
    key 0
    weight 1
  ]
  edge [
    source 3
    target 1
    key 0
    weight 1
  ]
  edge [
    source 3
    target 10
    key 0
    weight 1
  ]
  edge [
    source 4
    target 5
    key 0
    weight 1
  ]
  edge [
    source 5
    target 4
    key 0
    weight 1
  ]
  edge [
    source 7
    target 9
    key 0
    weight 1
  ]
  edge [
    source 8
    target 2
    key 0
    weight 1
  ]
  edge [
    source 9
    target 7
    key 0
    weight 1
  ]
  edge [
    source 10
    target 3
    key 0
    weight 1
  ]
]
