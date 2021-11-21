import flask
import pickle as pkl
import numpy as np

app = flask.Flask(__name__)

def ada_predict(arr):
    with open('adaboost_model.pkl', 'rb') as f:
        model = pkl.load(f)
        pred = model.predict(arr)[0]
        return pred

def tree_predict(arr):
    with open('decisiontree_model.pkl', 'rb') as f:
        model = pkl.load(f)
        pred = model.predict(arr)[0]
        return pred

def boost_predict(arr):
    with open('gradient_boost_model.pkl', 'rb') as f:
        model = pkl.load(f)
        pred = model.predict(arr)[0]
        return pred

def forest_predict(arr):
    with open('random_forest_model.pkl', 'rb') as f:
        model = pkl.load(f)
        pred = model.predict(arr)[0]
        return pred

@app.route('/', methods=['POST', 'GET'])
def predict():
    if flask.request.method == 'POST':
        prio = flask.request.form['prio']
        static_prio = flask.request.form['static_prio']
        free_area_cache = flask.request.form['free_area_cache']
        map_count = flask.request.form['map_count']
        total_vm = flask.request.form['total_vm']
        shared_vm = flask.request.form['shared_vm']
        reserved_vm = flask.request.form['reserved_vm']
        min_flt = flask.request.form['min_flt']
        fs_excl_counter = flask.request.form['fs_excl_counter']
        stime = flask.request.form['stime']
        gtime = flask.request.form['gtime']
        inp = np.array([prio, static_prio, free_area_cache, map_count, total_vm, shared_vm, reserved_vm, min_flt, fs_excl_counter, stime, gtime], dtype='float32')
        inp = inp.reshape(1, -1)
        ada_pred = ada_predict(inp)
        tree_pred = tree_predict(inp)
        boost_pred = boost_predict(inp)
        forest_pred = forest_predict(inp)
        if ada_pred + tree_pred + boost_pred + forest_pred > 1:
            final = 'MALWARE'
        else:
            final = 'BENIGN'
        return flask.render_template('index.html', final=final)
    return flask.render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
