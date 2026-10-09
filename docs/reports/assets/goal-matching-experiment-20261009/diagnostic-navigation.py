import json,os,sys,time
from pathlib import Path
from grutopia_extension.interactive_navigation.semantic_exploration_component import SemanticExplorationComponent
from grutopia.demo.go2_semantic_exploration import main

root=Path(os.environ['INTER_NAV_EXPERIMENT_ROOT'])
original=SemanticExplorationComponent.action
def action(self,step,observation):
    result=original(self,step,observation)
    if step%24==0:
        row={'step':step,'position':[float(v) for v in observation['position']],'orientation':[float(v) for v in observation['orientation']],'action':result,'goal':self.current_goal,'state':self.state,'active_waypoint':self.mapping.statistics().get('active_waypoint')}
        with (root/'actions.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
    return result
SemanticExplorationComponent.action=action
result=main()
print(json.dumps({'event':'diagnostic_main_return','return_code':result}),flush=True)
sys.exit(result)
