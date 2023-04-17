using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

public class MonteCarlo : MonoBehaviour
{
    public Animator animator;
    private int maxEpisodes = 500;
    private float stepCost = -0.1f;
    private float maxSteps = 1000000;
    private float discountFactor = 0.99f;
    private float cumulativeReward = 0f;
    private float stepSize = 1f;
    private List<Trajectory> trajectory = new List<Trajectory>();
    private Dictionary<string, Dictionary<int, float>> stateActionValues = new Dictionary<string, Dictionary<int, float>>();
    private Dictionary<string, Dictionary<int, List<float>>> returnValues = new Dictionary<string, Dictionary<int, List<float>>>();
    private List<Transform> obstacles = new List<Transform>();
    private float gridPositionConstant = 7f;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private int episodeCount = 0;
    private int steps = 0;
    private float targetThreshold = 0.5f;
    private bool isSavedData = false;
    private List<string> visitedStates = new List<string>();


    // Start is called before the first frame update
    void Start()
    {
        animator = GetComponent<Animator>();
        animator.SetBool("is_idle", false);
        startPosition = new Vector3(gridPositionConstant, 0.2f, -gridPositionConstant);
        targetPosition = new Vector3(-gridPositionConstant, 0.2f, gridPositionConstant);

        Dictionary<int, float> actionDict = new Dictionary<int, float>();
        actionDict.Add(0, 1f);
        actionDict.Add(1, 1f);
        actionDict.Add(2, 1f);
        actionDict.Add(3, 1f);
        stateActionValues.Add(getState(targetPosition), actionDict);

        GameObject[] obstacleObjects = GameObject.FindGameObjectsWithTag("Obstacle");
        foreach (GameObject obstacleObject in obstacleObjects) {
            obstacles.Add(obstacleObject.transform);
        }

        if (SaveStateActionData.retrieve() != null) {
            stateActionValues = SaveStateActionData.retrieve();
            isSavedData = true;
            maxEpisodes = 1;
            Application.targetFrameRate = 10;
            print("Using saved values!");
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (episodeCount == maxEpisodes) {
            if (!isSavedData) {
                SaveStateActionData.save(stateActionValues);
            }
            return;
        }

        if (isSavedData) {
            if (!(Mathf.Abs(transform.position.x - targetPosition.x) <= targetThreshold && Mathf.Abs(transform.position.z - targetPosition.z) <= targetThreshold)) {
                int action = chooseAction();
                visitedStates.Add(getState(transform.position));
                move(action);
                steps += 1;
            } else {
                animator.SetBool("is_idle", true);
                print("Steps taken = " + steps);
            }
        } else {
            runMonteCarloMethod();
        }
    }

    private async void runMonteCarloMethod() {
        Vector3 previousPos = transform.position;
        int action = chooseAction();
        move(action);
        steps += 1;
        float reward = stepCost;
        if ((Mathf.Abs(transform.position.x - targetPosition.x) <= targetThreshold && Mathf.Abs(transform.position.z - targetPosition.z) <= targetThreshold) || steps >= maxSteps) {
            reward = (steps >= maxSteps) ? steps*stepCost : 0f;
        }
        string state = getState(previousPos);
        Trajectory t = new Trajectory(state, action, reward);
        trajectory.Add(t);
        
        if ((Mathf.Abs(transform.position.x - targetPosition.x) <= targetThreshold && Mathf.Abs(transform.position.z - targetPosition.z) <= targetThreshold) || steps >= maxSteps) {
            for (int i = trajectory.Count - 1; i >= 0; i--) {
                Trajectory currentTrajectory = trajectory[i];
                cumulativeReward = discountFactor * cumulativeReward + currentTrajectory.reward;
                int endRange = i <= 0 ? 0 : i-1;
                if (!trajectory.GetRange(0, endRange).Any(t => t.state == currentTrajectory.state && t.action == currentTrajectory.action)) {
                    addToReturnValues(currentTrajectory.state, currentTrajectory.action, cumulativeReward);
                    addToStateActionValues(currentTrajectory);
                }
            }

            resetEpisode();
        }
    }

    private void resetEpisode() {
        print("Episode = " + episodeCount + "Steps = " + steps);
        steps = 0;
        episodeCount += 1;
        transform.position = startPosition;
        cumulativeReward = 0f;
        trajectory.Clear();
    }

    private void addToReturnValues(string state, int action, float value) {
        if (returnValues.ContainsKey(state) && returnValues[state].ContainsKey(action)) {
            returnValues[state][action].Add(value);
        }
        else {
            if (!returnValues.ContainsKey(state)) {
                returnValues.Add(state, new Dictionary<int, List<float>>());
            }
            if (!returnValues[state].ContainsKey(action)) {
                returnValues[state].Add(action, new List<float>());
            }
            returnValues[state][action].Add(value);
        }
    }

    private void addToStateActionValues(Trajectory currentTrajectory) {
        string state = currentTrajectory.state;
        int action = currentTrajectory.action;

        if (returnValues.ContainsKey(state) && returnValues[state].ContainsKey(action)) {
            float mean = returnValues[state][action].Average();
            if (stateActionValues.ContainsKey(state)) {
                stateActionValues[state][action] = mean;
            }
            else {
                stateActionValues.Add(state, new Dictionary<int, float> {{action, mean}});
            }
        }
    }

    private string getState(Vector3 position)
    {
        int x = Mathf.RoundToInt(position.x);
        int z = Mathf.RoundToInt(position.z);
        return x + "," + z;
    }

    protected int chooseAction() {
        // Character: 0 -> forwward, 1 -> backward, 2 -> right, 3 -> left
        List<int> actions = new List<int>() {0, 1, 2, 3};
        if (transform.position.x >= gridPositionConstant && transform.position.z >= gridPositionConstant) {
            actions.Remove(2);
            actions.Remove(0);
        } else if (transform.position.x <= -gridPositionConstant && transform.position.z <= -gridPositionConstant) {
            actions.Remove(3);
            actions.Remove(1);
        } else if (transform.position.x >= gridPositionConstant && transform.position.z <= -gridPositionConstant) {
            actions.Remove(2);
            actions.Remove(1);
        } else if (transform.position.x <= -gridPositionConstant && transform.position.z >= gridPositionConstant) {
            actions.Remove(3);
            actions.Remove(0);
        } else if (transform.position.x >= gridPositionConstant) {
            actions.Remove(2);
        } else if (transform.position.z >= gridPositionConstant) {
            actions.Remove(0);
        } else if (transform.position.x <= -gridPositionConstant) {
            actions.Remove(3);
        } else if (transform.position.z <= -gridPositionConstant) {
            actions.Remove(1);
        }

        if (!isSavedData) {
            int index = Random.Range(0, actions.Count);
            return actions[index];
        } else {
            return findBestAction(actions);
        }
    }

    private int findBestAction(List<int> actions) {
        string state = getState(transform.position);
        if (stateActionValues.ContainsKey(state)) {
            Dictionary<int, float> stateValues = stateActionValues[state];
            int maxKey = 5;
            float maxValue = float.MinValue;
            foreach (KeyValuePair<int, float> pair in stateValues) {
                if (pair.Value > maxValue && actions.Contains(pair.Key) && !visitedStates.Contains(getState(newPos(pair.Key)))) {
                    maxValue = pair.Value;
                    maxKey = pair.Key;
                }
            }
            if (maxKey == 5) {
                return Random.Range(0, actions.Count);
            }
            
            return maxKey;
        }
        return Random.Range(0, actions.Count);
    }

    protected void move(int action) {
        // Character: 0 -> forwward, 1 -> backward, 2 -> right, 3 -> left
        Vector3 newPosition = transform.position;
        
        Vector3 direction = transform.forward;
        switch (action) {
            case 0:
                direction = Vector3.forward;
                newPosition.z += stepSize;
                break;
            case 1:
                direction = -Vector3.forward;
                newPosition.z -= stepSize;
                break;
            case 2:
                direction = Vector3.right;
                newPosition.x += stepSize;
                break;
            case 3:
                direction = -Vector3.right;
                newPosition.x -= stepSize;
                break;
        }
        Quaternion newRotation = Quaternion.LookRotation(direction, transform.up);
        newPosition = new Vector3(Mathf.Clamp(newPosition.x, -gridPositionConstant, gridPositionConstant), newPosition.y, Mathf.Clamp(newPosition.z, -gridPositionConstant, gridPositionConstant));

        if (!isSavedData) {
            if (distanceToNearestObstacle(newPosition, newRotation) < 0.5 ) {
                return;
            }
        }
        
        transform.rotation = newRotation;
        transform.position = newPosition;
    }

    protected Vector3 newPos(int action) {
        Vector3 newPosition = transform.position;
        
        Vector3 direction = transform.forward;
        switch (action) {
            case 0:
                direction = Vector3.forward;
                newPosition.z += stepSize;
                break;
            case 1:
                direction = -Vector3.forward;
                newPosition.z -= stepSize;
                break;
            case 2:
                direction = Vector3.right;
                newPosition.x += stepSize;
                break;
            case 3:
                direction = -Vector3.right;
                newPosition.x -= stepSize;
                break;
        }
        newPosition = new Vector3(Mathf.Clamp(newPosition.x, -gridPositionConstant, gridPositionConstant), newPosition.y, Mathf.Clamp(newPosition.z, -gridPositionConstant, gridPositionConstant));
        return newPosition;
    }

    protected float distanceToNearestObstacle(Vector3 position, Quaternion rotation) {
        float minDistance = float.MaxValue;
        foreach (Transform obstacle in obstacles) {
            Vector3 directionToObstacle = obstacle.position - position;
            float angle = Vector3.Angle(directionToObstacle, rotation * Vector3.forward);
            if (angle < 90.0f) {
                float distance = directionToObstacle.magnitude;
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
        }
        return minDistance;
    }

    private struct Trajectory {
        public string state;
        public int action;
        public float reward;

        public Trajectory(string state, int action, float reward) {
            this.state = state;
            this.action = action;
            this.reward = reward;
        }
    }
}

public static class SaveStateActionData {
    private static string filePath = Application.persistentDataPath + "/" + "MonteCarlo.dat";
    public static void save(Dictionary<string, Dictionary<int, float>> stateActionValues) {
        BinaryFormatter formatter = new BinaryFormatter();

        using (FileStream stream = new FileStream(filePath, FileMode.Create))
        {
            formatter.Serialize(stream, stateActionValues);
        }
    }

    public static Dictionary<string, Dictionary<int, float>> retrieve() {
        Dictionary<string, Dictionary<int, float>> stateActionValues = new Dictionary<string, Dictionary<int, float>>();
        if (File.Exists(filePath))
        {
            BinaryFormatter formatter = new BinaryFormatter();
            List<string> str = new List<string>();

            using (FileStream stream = new FileStream(filePath, FileMode.Open))
            {
                stateActionValues = (Dictionary<string, Dictionary<int, float>>)formatter.Deserialize(stream);
            }
        }
        else
        {
            return null;
        }

        return stateActionValues;
    }
}
