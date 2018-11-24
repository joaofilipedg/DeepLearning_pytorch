import numpy as np

def viterbi_log (observations, obs_list, states, start_probs, stop_probs, trans_probs, emiss_probs, days):
    num_states = len(states)
    num_observations = len(observations)

    st_seq = np.zeros(shape=(num_observations), dtype=np.int)
    Vit = np.zeros(shape=(num_observations, num_states), dtype=np.float)
    Backtrack = np.zeros(shape=(num_observations, num_states), dtype=np.float)

    #initialization
    print('Observation 0: %s observed %s' %(days[0], obs_list[observations[0]]))
    for state in range(0,num_states):
        Vit[0,state] = np.log(start_probs[state]) + np.log(emiss_probs[state,observations[0]])
        print ('OBS 0 : log(%.1f) + log(%.1f) = %.3f <--- %s' % (np.log(start_probs[state]), np.log(emiss_probs[state,observations[0]]), Vit[0][state], states[state]))

    #forward
    for obs in range(1, num_observations):
        print('Observation %d: %s observed %s' %(obs, days[obs], obs_list[observations[obs]]))
        for state in range(0,num_states):
            prod_aux = [np.log(trans_probs[state,s2]) + (Vit[obs-1,s2]) for s2 in range(0,num_states)]

            Vit[obs,state] = np.max(prod_aux) + np.log(emiss_probs[state,observations[obs]])
            print ('OBS %d : log(%.1f) + log(%.1f)) = %.3f <--- %s' % (obs,  np.max(prod_aux), np.log(emiss_probs[state,observations[obs]]), Vit[obs,state], states[state]))

            Backtrack[obs,state] = np.argmax(prod_aux)

    #Backward
    prod_aux_2 = [np.log(stop_probs[s2]) + Vit[num_observations-1,s2] for s2 in range(0,num_states)]
    st_seq[num_observations-1] = np.argmax(prod_aux_2)
    for obs in range(num_observations-2, -1, -1):
        st_seq[obs] = Backtrack[obs+1,st_seq[obs+1]]

    return st_seq, Vit

def viterbi (observations, obs_list, states, start_probs, stop_probs, trans_probs, emiss_probs, days):
    num_states = len(states)
    num_observations = len(observations)

    st_seq = np.zeros(shape=(num_observations), dtype=np.int)
    Vit = np.zeros(shape=(num_observations, num_states), dtype=np.float)
    Backtrack = np.zeros(shape=(num_observations, num_states), dtype=np.float)

    #initialization
    print('Observation 0: %s observed %s' %(days[0], obs_list[observations[0]]))
    for state in range(0,num_states):
        Vit[0,state] = start_probs[state] * emiss_probs[state,observations[0]]
        print ('OBS 0 : %.1f * %.1f = %.4f <--- %s' % (start_probs[state], emiss_probs[state,observations[0]], Vit[0,state], states[state]))

    #forward
    for obs in range(1, num_observations):
        print('Observation %d: %s observed %s' %(obs, days[obs], obs_list[observations[obs]]))
        for state in range(0,num_states):
            prod_aux = [trans_probs[state,s2] * Vit[obs-1,s2] for s2 in range(0,num_states)]

            Vit[obs,state] = np.max(prod_aux) * emiss_probs[state,observations[obs]]
            print ('OBS %d : %.4f * %.1f = %.8f <--- %s' % (obs, np.max(prod_aux), emiss_probs[state,observations[obs]], Vit[obs,state], states[state]))

            Backtrack[obs,state] = np.argmax(prod_aux)

    #Backward
    prod_aux_2 = [stop_probs[s2] * Vit[num_observations-1,s2] for s2 in range(0,num_states)]
    st_seq[num_observations-1] = np.argmax(prod_aux_2)
    for obs in range(num_observations-2, -1, -1):
        st_seq[obs] = Backtrack[obs+1,st_seq[obs+1]]

    return st_seq, Vit

def forward_backward (observations, obs_list, states, start_probs, stop_probs, trans_probs, emiss_probs, days):
    num_states = len(states)
    num_observations = len(observations)

    Forward = np.zeros(shape=(num_observations, num_states), dtype=np.float)
    Backward = np.zeros(shape=(num_observations, num_states), dtype=np.float)

    #forward
    for state in range(0,num_states):
        Forward[0,state] =  start_probs[state] * emiss_probs[state,observations[0]]
    for obs in range(1, num_observations):
        for state in range(0,num_states):
            sum_aux = np.sum([trans_probs[state, s2] * Forward[obs-1, s2] for s2 in range(0,num_states)])
            Forward[obs,state] =  sum_aux * emiss_probs[state,observations[obs]]

    #backward
    for state in range(0,num_states):
        Backward[num_observations-1,state] =  stop_probs[state]
    for obs in range(num_observations-2, -1, -1):
        for state in range(0,num_states):
            Backward[obs,state] = np.sum([trans_probs[s2,state] * Backward[obs+1,s2] * emiss_probs[s2,observations[obs+1]] for s2 in range(0,num_states)])

    return Forward, Backward
#
def main():
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weathers = ['Sunny', 'Windy', 'Rainy']
    activities = ['Surf', 'Beach', 'Videogame', 'Study']

    start_probability = [0.2, 0.3, 0.5]
    stop_probability = [0.7, 0.3, 0.2]

    num_weathers = len(weathers)
    num_activities = len(activities)

    transition_probability = np.zeros(shape=(num_weathers,num_weathers), dtype=np.float)
    transition_probability[0] = [0.7, 0.3, 0.2]
    transition_probability[1] = [0.2, 0.5, 0.3]
    transition_probability[2] = [0.1, 0.2, 0.5]

    emission_probability = np.zeros(shape=(num_weathers,num_activities), dtype=np.float)
    emission_probability[0] = [0.4, 0.4, 0.1, 0.1]
    emission_probability[1] = [0.5, 0.1, 0.2, 0.2]
    emission_probability[2] = [0.1, 0.1, 0.3, 0.5]

    observations = [2, 3, 3, 0, 1, 2, 1]
    num_observations = len(observations)

    print('\nViterbi in Logs:\n')
    (states_seq_log, V_log) = viterbi_log(observations, activities, weathers, start_probability, stop_probability, transition_probability, emission_probability, days)

    print('V_log matrix:')
    print(V_log)
    print('\nViterbi_log predictions for the week:')
    for obs in range(0, num_observations):
        print('%s predicted %s' %(days[obs], weathers[states_seq_log[obs]]))

    print('\nForward-Backward:\n')
    (Forward, Backward) = forward_backward(observations, activities, weathers, start_probability, stop_probability, transition_probability, emission_probability, days)
    print('Forward Matrix:')
    print(Forward)
    print('\nBackward Matrix:')
    print(Backward)

    #Sanity Check
    print('\nSanity check (P(X=x) for all observations):')
    for obs in range(0, num_observations):
        P_X = np.sum([Forward[obs,s2] * Backward[obs,s2] for s2 in range(0,num_weathers)])
        print(P_X)

    #State posteriors
    P_y_x = np.zeros(shape=(num_observations, num_weathers), dtype=np.float)
    states_seq_bf = np.zeros(shape=(num_observations), dtype=np.int)
    for obs in range(0, num_observations):
        for state in range(0, num_weathers):
            P_y_x[obs, state] = Forward[obs, state] * Backward[obs, state]
        states_seq_bf[obs] = np.argmax(P_y_x[obs,:])

    print('\nBackward-forward predictions for the week:')
    for obs in range(0, num_observations):
        print('%s predicted %s' %(days[obs], weathers[states_seq_bf[obs]]))

if __name__ == "__main__":
    main()
