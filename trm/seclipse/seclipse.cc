//
// Python/C++ interface file for seclipse to do the hard work
// TBD
//

#include <Python.h>
#include "numpy/arrayobject.h"
#include "trm_constants.h"
#include "trm_subs.h"


// Adds up optical depth due to lines

static PyObject* 
integ(PyObject *self, PyObject *args, PyObject *kwords)
{
  
    // Process and check arguments
    PyObject *wave  = NULL;
    std::vector<Line> lines;
    double temp, nelec;
    std::vector<Elem> elems;
    PyObject *amass  = NULL;
    double acc=1.e-8;

    static const char *kwlist[] = {"s1", "r1", "limb1", "n1", "r2", "p", NULL};

??????????

    if(!PyArg_ParseTupleAndKeywords(args, kwords, "OO&ddO&O|d", (char**)kwlist, &wave, 
				    lineconv, (void*)&lines, &temp, &nelec, 
				    elemconv, (void*)&elems, &amass, &acc))
	return NULL;

    // check inputs
    if(PyArray_Check(wave)){
	int nd = PyArray_NDIM(wave);
	if(nd != 1){
	    PyErr_SetString(PyExc_ValueError, "atomic.addlines: wave must be a 1D array");
	    return NULL;
	}
    }else{
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: wave must be an array");
	return NULL;
    }

    if(temp <= 0.){
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: temp <= 0");
	return NULL;
    }

    if(nelec <= 0.){
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: nelec <= 0");
	return NULL;
    }

    if(PyArray_Check(amass)){
	int nd = PyArray_NDIM(amass);
	if(nd != 1){
	    PyErr_SetString(PyExc_ValueError, "atomic.addlines: amass must be a 1D array");
	    return NULL;
	}
    }else{
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: amass must be an array");
	return NULL;
    }

    if(acc <= 0.){
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: acc <= 0");
	return NULL;
    }

    // get pointer to wavelengths
    npy_intp nwave = PyArray_Size(wave);
    if(!nwave) {
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: wave array must have at least one element");
	return NULL;
    }
    PyObject *twave = PyArray_FROM_OTF(wave, NPY_DOUBLE, NPY_IN_ARRAY);
    if(twave == NULL){
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: failed to make a double array from wave");
	return NULL;
    }
    double *dwave = (double *)PyArray_DATA(twave);

    // get pointer to atomic masses
    npy_intp namass = PyArray_Size(amass);
    if(!namass) {
	Py_XDECREF(twave);
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: amass array must have at least one element");
	return NULL;
    }
    PyObject *tamass = PyArray_FROM_OTF(amass, NPY_DOUBLE, NPY_IN_ARRAY);
    if(tamass == NULL){
	Py_XDECREF(twave);
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: failed to make a double array from amass");
	return NULL;
    }
    double *damass = (double *)PyArray_DATA(tamass);

    // Create array to hold results
    npy_intp dim[1] = {nwave};
    PyArrayObject* oarr = (PyArrayObject*)PyArray_SimpleNew(1, dim, PyArray_DOUBLE);
    if(oarr == NULL){
	PyErr_SetString(PyExc_ValueError, "atomic.addlines: failed to create the output array");
	Py_XDECREF(tamass);
	Py_XDECREF(twave);
	return NULL;
    }
    double *oarrp = (double*)oarr->data;    

    // finally ... do something

    // zero output
    for(npy_intp i=0; i<nwave; i++)
	oarrp[i] = 0.;

    // go through line by line
    unsigned long int jlo = nwave/2, jhi, lnwave = nwave;

    // number of sigma either side of line centre to go
    const double NRMS = 10.;

    // Next is usual "pi*e^2/(me*c)" factor corrected to SI (units m^2 Hz) 
    // The 1.e-7 is just \mu_0/4\pi
    const double AREA = 1.e-7*Constants::PI*std::pow(Constants::E,2)*Constants::C/Constants::ME;
    const double STIM = Constants::H*Constants::C*1.e9/Constants::K/temp;
    const double RTPI = std::sqrt(2.*Constants::PI);

    for(size_t i=0; i<lines.size(); i++){

	Line&  line = lines[i];

	if(line.elem >= namass){
	    PyErr_SetString(PyExc_ValueError, "atomic.addlines: element out of range of supplied atomic masses");
	    Py_XDECREF(tamass);
	    Py_XDECREF(twave);
	    return NULL;
	}
	if(line.elem >= elems.size()){
	    PyErr_SetString(PyExc_ValueError, "atomic.addlines: element out of range of supplied Elem list");
	    Py_XDECREF(tamass);
	    Py_XDECREF(twave);
	    return NULL;
	}

	Elem&  elem = elems[line.elem];

	// define lower and upper wavelength limits.
	// vdopp = RMS line-of-sight speed, m/s
	// f = vdopp / C
	double vdopp = std::sqrt(Constants::K*temp/Constants::MP/damass[line.elem]);
	double dlam  = line.wave * vdopp / Constants::C;
	double wlo   = line.wave - NRMS*dlam;
	double whi   = line.wave + NRMS*dlam;

	// find corresponding pixel limits
	Subs::hunt(dwave, lnwave, wlo, jlo);
	if(jlo >= lnwave) break;

	jhi = jlo;
	Subs::hunt(dwave, lnwave, whi, jhi);
	
	if(jhi > jlo){

	    // Compute optical depth at line centre, corrected
	    // for stimulated emission
	    double strength = AREA*elem.wdens[line.ion-elem.lstate]*
		std::pow(10.,line.lgf)*std::exp(-line.enlo/temp)/
		(RTPI*1.e9*vdopp/line.wave)*(1.0-std::exp(-STIM/line.wave));

	    // finally add into array
	    for(unsigned int j=jlo; j<jhi; j++)
		oarrp[j] += strength*exp(-pow((dwave[j]-line.wave)/dlam, 2)/2.);
	}
    }

    Py_XDECREF(tamass);
    Py_XDECREF(twave);
    
    return Py_BuildValue("N", oarr);
};

// The methods

static PyMethodDef OrbitsMethods[] = {

    {"addlines", (PyCFunction)atomic_addlines, METH_VARARGS | METH_KEYWORDS, 
     "addlines(wave, lines, temp, nelec, elems, acc=1.e-8),\n\n"
     "computes opacity from atomic lines\n\n"
     "Arguments:\n\n"
     "  wave     -- array of wavelengths to compute opacities for.\n"
     "  lines    -- list of Line objects containing atomic line data\n"
     "  temp     -- temperature (K)\n"
     "  nelec    -- electron density (/m^3)\n"
     "  elems    -- list of Elem objects containing weighted ion column densities\n"
     "  acc      -- fractional accuracy parameter\n\n"
     "Returns: opacity -- array of opacities\n\n"
    },

    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
init_atomic(void)
{
    (void) Py_InitModule("_atomic", OrbitsMethods);
    import_array();
}
