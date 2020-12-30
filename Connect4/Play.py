import torch
from Game import Connect4
from DNN import DNN

dnn = DNN()
dnn.load_state_dict(torch.load('Connect4_Agent.pth'))
game = Connect4()
state = game.init_state
end = False
player = 1
game.plot(state)

while not end:
    a = torch.argmax(dnn(state)[0])
    state = game.play(state, a, player)
    end, _ = game.check_end(state)
    game.plot(state)
    player *= -1
    
    
    if not end:
        a = int(input())
        state = game.play(state, a, player)
        end, _ = game.check_end(state)
        game.plot(state)
        player *= -1
