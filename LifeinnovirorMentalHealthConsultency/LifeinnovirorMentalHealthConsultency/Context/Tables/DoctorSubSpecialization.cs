using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class DoctorSubSpecialization
    {
        [ForeignKey("Doctor")]
        public int DoctorId { get; set; }
        public Doctor Doctor { get; set; }

        [ForeignKey("SubSpecialization")]
        public int SubSpecializationId { get; set; }
        public SubSpecialization SubSpecialization { get; set; }
    }
}